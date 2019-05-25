package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/pkg/errors"

	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var Float = tensor.Float32

type contextualError interface {
	error
	Node() *Node
	Value() Value
	InstructionID() int
	Err() error
}

// WHERE TO 1HV THE INPUITS?????
// func OneHotVector(id, classes int, t tensor.Dtype, opts ...NodeConsOpt) *Node {

// GRU is a standard GRU node. Geddit?
type GRU struct {

	// weights for mem
	u *Node
	w *Node
	b *Node

	// update gate
	uz *Node
	wz *Node
	bz *Node

	// reset gate
	ur  *Node
	wr  *Node
	br  *Node
	one *Node

	Name string // optional name
}

func MakeGRU(name string, g *ExprGraph, inputSize, hiddenSize int, dt tensor.Dtype) GRU {
	// standard weights
	u := NewMatrix(g, dt, WithShape(hiddenSize, hiddenSize), WithName(fmt.Sprintf("%v.u", name)), WithInit(GlorotN(1.0)))
	w := NewMatrix(g, dt, WithShape(hiddenSize, inputSize), WithName(fmt.Sprintf("%v.w", name)), WithInit(GlorotN(1.0)))
	b := NewVector(g, dt, WithShape(hiddenSize), WithName(fmt.Sprintf("%v.b", name)), WithInit(Zeroes()))

	// update gate
	uz := NewMatrix(g, dt, WithShape(hiddenSize, hiddenSize), WithName(fmt.Sprintf("%v.uz", name)), WithInit(GlorotN(1.0)))
	wz := NewMatrix(g, dt, WithShape(hiddenSize, inputSize), WithName(fmt.Sprintf("%v.wz", name)), WithInit(GlorotN(1.0)))
	bz := NewVector(g, dt, WithShape(hiddenSize), WithName(fmt.Sprintf("%v.b_uz", name)), WithInit(Zeroes()))

	// reset gate
	ur := NewMatrix(g, dt, WithShape(hiddenSize, hiddenSize), WithName(fmt.Sprintf("%v.ur", name)), WithInit(GlorotN(1.0)))
	wr := NewMatrix(g, dt, WithShape(hiddenSize, inputSize), WithName(fmt.Sprintf("%v.wr", name)), WithInit(GlorotN(1.0)))
	br := NewVector(g, dt, WithShape(hiddenSize), WithName(fmt.Sprintf("%v.bz", name)), WithInit(Zeroes()))

	ones := tensor.Ones(dt, hiddenSize)
	one := g.Constant(ones)
	gru := GRU{
		u: u,
		w: w,
		b: b,

		uz: uz,
		wz: wz,
		bz: bz,

		ur: ur,
		wr: wr,
		br: br,

		one: one,
	}
	return gru
}

func (l *GRU) Activate(x, prev *Node) (retVal *Node, err error) {
	// update gate
	// z := Must(Sigmoid(Must(Add(Must(Add(Must(Mul(l.uz, prev)), Must(l.wz, x))), l.bz))))
	uzh := Must(Mul(l.uz, prev))
	wzx := Must(Mul(l.wz, x))
	z := Must(Sigmoid(
		Must(Add(
			Must(Add(uzh, wzx)),
			l.bz))))

	// reset gate
	// r := Must(Sigmoid(Must(Add(Must(Add(Must(Mul(l.wr, x)), Must(Mul(l.ur, prev)), l.br))))))
	urh := Must(Mul(l.ur, prev))
	wrx := Must(Mul(l.wr, x))
	r := Must(Sigmoid(
		Must(Add(
			Must(Add(urh, wrx)),
			l.br))))

	// memory for hidden
	hiddenFilter := Must(Mul(l.u, Must(HadamardProd(r, prev))))
	wx := Must(Mul(l.w, x))
	mem := Must(Tanh(
		Must(Add(
			Must(Add(hiddenFilter, wx)),
			l.b))))

	omz := Must(Sub(l.one, z))
	omzh := Must(HadamardProd(omz, prev))
	upd := Must(HadamardProd(z, mem))
	retVal = Must(Add(upd, omzh))
	return
}

func (l *GRU) learnables() Nodes {
	retVal := make(Nodes, 0, 9)
	retVal = append(retVal, l.u, l.w, l.b, l.uz, l.wz, l.bz, l.ur, l.wr, l.br)
	return retVal
}

type seq2seq struct {
	in        GRU
	dummyPrev *Node // (hiddnsize) vector
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
	in := MakeGRU("In", g, embSize, hiddenSize, Float)

	dummyPrev := NewVector(g, Float, WithShape(hiddenSize), WithName("Dummy Prev"), WithInit(Zeroes()))
	embedding := NewMatrix(g, Float, WithShape(len(vocab), embSize), WithInit(GlorotN(1.0)), WithName("Embedding"))
	decoder := NewMatrix(g, Float, WithShape(len(vocab), hiddenSize), WithInit(GlorotN(1.0)), WithName("Output Decoder"))
	// dummyPrev2 := NewVector(g, Float, WithShape(hiddenSize), WithName("Dummy Prev2"), WithInit(Zeroes()))
	// keyEmbedding := NewMatrix(g, Float, WithShape(keySize, embSize), WithName("Key Embedding"), WithInit(GlorotN(1.0)))
	// durEmbedding := NewMatrix(g, Float, WithShape(durationSize, embSize), WithName("Duration Embedding"), WithInit(GlorotN(1.0)))

	// the reason for 3xembSize:
	// each entry (key, dur) has embsize
	// furthermore, there is an interaction variable (which is the hadamard prod of both entries).

	return &seq2seq{
		in:        in,
		dummyPrev: dummyPrev,
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
	var prev *Node = s.dummyPrev
	s.predvals = make([]Value, maxsent)
	for i := 0; i < maxsent; i++ {
		vec := Must(Slice(s.embedding, S(0))) // dummy, to be replaced at runtime
		s.inVecs = append(s.inVecs, vec)
		if prev, err = s.in.Activate(vec, prev); err != nil {
			return
		}
		prediction := Must(SoftMax(Must(Mul(s.decoder, prev))))
		s.preds = append(s.preds, prediction)
		Read(prediction, &s.predvals[i])
		loss := Must(Slice(prediction, S(0))) // dummy, to be replaced at runtime
		s.losses = append(s.losses, prediction)

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

		srcID := vocabIndex[currentRune]
		targetID := vocabIndex[correctPrediction]
		Let(s.inVecs[i], S(srcID))
		Let(s.losses[i], S(targetID))
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
	vm := NewTapeMachine(g2, TraceExec())
	if err = vm.RunAll(); err != nil {
		return
	}
	defer vm.Close()
	for _, pred := range s.predvals {
		id := sample(pred)
		r := vocab[id]
		if r == START {
			continue
		}
		if r == END || r == BLANK {
			break
		}

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
		fmt.Printf("Cost for epoch %d: %1.10f\n", e, costVal)
	}

	return nil

}
