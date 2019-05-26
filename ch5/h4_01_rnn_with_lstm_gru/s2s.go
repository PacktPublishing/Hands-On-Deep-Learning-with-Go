package main

import (
	"encoding/json"
	"io"
	"log"
	"os"

	"github.com/pkg/errors"
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type seq2seq struct {
	in        LSTM
	dummyPrev *Node // (???) vector
	dummyCell *Node // (??? ) vector
	embedding *Node // NxM matrix, where M is the number of dimensions of the embedding

	decoder *Node
	vocab   []rune

	inVecs   []*Node
	losses   []*Node
	preds    []*Node
	predvals []Value
	g        *ExprGraph
	vm       VM
}

// NewS2S creates a new Seq2Seq network. Input size is the size of the embedding. Hidden size is the size of the hidden layer
func NewS2S(hiddenSize, embSize int, vocab []rune) *seq2seq {
	g := NewGraph()
	// in := MakeGRU("In", g, embSize, hiddenSize, Float)s
	in := MakeLSTM(g, hiddenSize, embSize)
	log.Printf("%q", vocab)

	dummyPrev := NewVector(g, Float, WithShape(embSize), WithName("Dummy Prev"), WithInit(Zeroes()))
	dummyCell := NewVector(g, Float, WithShape(hiddenSize), WithName("Dummy Cell"), WithInit(Zeroes()))
	embedding := NewMatrix(g, Float, WithShape(len(vocab), embSize), WithInit(GlorotN(1.0)), WithName("Embedding"))
	decoder := NewMatrix(g, Float, WithShape(len(vocab), hiddenSize), WithInit(GlorotN(1.0)), WithName("Output Decoder"))

	return &seq2seq{
		in:        in,
		dummyPrev: dummyPrev,
		dummyCell: dummyCell,
		embedding: embedding,
		vocab:     vocab,
		decoder:   decoder,
		g:         g,
	}
}

func (s *seq2seq) learnables() Nodes {
	retVal := make(Nodes, 0)
	retVal = append(retVal, s.in.learnables()...)
	retVal = append(retVal, s.embedding)
	retVal = append(retVal, s.decoder)
	return retVal
}

func (s *seq2seq) build() (cost *Node, err error) {
	// var prev *Node = s.dummyPrev
	prev := lstmout{
		hidden: s.dummyCell,
		cell:   s.dummyCell,
	}
	s.predvals = make([]Value, maxsent)

	var prediction *Node
	for i := 0; i < maxsent; i++ {
		var vec *Node
		if i == 0 {
			vec = Must(Slice(s.embedding, S(0))) // dummy, to be replaced at runtime
		} else {
			vec = Must(Mul(prediction, s.embedding))
		}
		s.inVecs = append(s.inVecs, vec)
		if prev, err = s.in.Activate(vec, prev); err != nil {
			return
		}
		prediction = Must(SoftMax(Must(Mul(s.decoder, prev.hidden))))
		s.preds = append(s.preds, prediction)
		Read(prediction, &s.predvals[i])

		logprob := Must(Neg(Must(Log(prediction))))
		loss := Must(Slice(logprob, S(0))) // dummy, to be replaced at runtime
		s.losses = append(s.losses, loss)

		if cost == nil {
			cost = loss
		} else {
			cost = Must(Add(cost, loss))
		}
	}

	_, err = Grad(cost, s.learnables()...)
	return
}

func (s *seq2seq) train(in []rune) (err error) {

	for i := 0; i < maxsent; i++ {
		var currentRune, correctPrediction rune
		switch {
		case i == 0:
			currentRune = START
			correctPrediction = in[i]
		case i-1 == len(in)-1:
			currentRune = in[i-1]
			correctPrediction = END
		case i-1 >= len(in):
			currentRune = BLANK
			correctPrediction = BLANK
		default:
			currentRune = in[i-1]
			correctPrediction = in[i]
		}

		targetID := vocabIndex[correctPrediction]
		if i == 0 || i-1 >= len(in) {
			srcID := vocabIndex[currentRune]
			UnsafeLet(s.inVecs[i], S(srcID))
		}
		UnsafeLet(s.losses[i], S(targetID))

	}
	if s.vm == nil {
		s.vm = NewTapeMachine(s.g, BindDualValues())
	}
	s.vm.Reset()
	err = s.vm.RunAll()

	return
}

func (s *seq2seq) predict(in []rune) (output []rune, err error) {
	g2 := s.g.SubgraphRoots(s.preds...)
	vm := NewTapeMachine(g2)
	if err = vm.RunAll(); err != nil {
		return
	}
	defer vm.Close()
	for _, pred := range s.predvals {
		log.Printf("%v", pred.Shape())
		id := sample(pred)
		if id >= len(vocab) {
			log.Printf("Predicted %d. Len(vocab) %v", id, len(vocab))
			continue
		}
		r := vocab[id]

		output = append(output, r)
	}
	return
}

func (s *seq2seq) checkpoint() (err error) {
	learnables := s.learnables()
	var f io.WriteCloser
	if f, err = os.OpenFile("CHECKPOINT.bin", os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644); err != nil {
		return
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, l := range learnables {
		t := l.Value().(*tensor.Dense).Data() // []float32
		if err = enc.Encode(t); err != nil {
			return
		}
	}

	return nil
}

func (s *seq2seq) load() (err error) {
	learnables := s.learnables()
	var f io.ReadCloser
	if f, err = os.OpenFile("CHECKPOINT.bin", os.O_RDONLY, 0644); err != nil {
		return
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	for _, l := range learnables {
		t := l.Value().(*tensor.Dense).Data().([]float32)
		var data []float32
		if err = dec.Decode(&data); err != nil {
			return
		}
		if len(data) != len(t) {
			return errors.Errorf("Unserialized length %d. Expected length %d", len(data), len(t))
		}
		copy(t, data)
	}
	return nil
}

func train(s *seq2seq, epochs int, solver Solver, data []string) (err error) {
	cost, err := s.build()
	if err != nil {
		return err
	}
	var costVal Value
	Read(cost, &costVal)

	model := NodesToValueGrads(s.learnables())
	for e := 0; e < epochs; e++ {
		shuffle(data)

		for _, sentence := range data {
			asRunes := []rune(sentence)
			if err = s.train(asRunes); err != nil {
				return
			}
			if err = solver.Step(model); err != nil {
				return
			}
		}
		// if e%100 == 0 {
		log.Printf("Cost for epoch %d: %1.10f\n", e, costVal)
		// }

	}

	return nil

}
