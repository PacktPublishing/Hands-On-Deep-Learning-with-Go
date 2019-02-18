package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/chewxy/math32"
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

func (l *GRU) learnables() []ValueGrad {
	retVal := make([]ValueGrad, 0, 9)
	retVal = append(retVal, l.u, l.w, l.b, l.uz, l.wz, l.bz, l.ur, l.wr, l.br)
	return retVal
}

type seq2seq struct {
	in           GRU
	in2          GRU
	dummyPrev    *Node // (hiddnsize) vector
	dummyPrev2   *Node
	keyEmbedding *Node // NxM matrix, where M is the number of dimensions of the embedding
	// chEmbedding  *Node // NxM matrix, where M is the number of dimensions of the embedding
	durEmbedding *Node // NxM matrix, where M is the number of dimensions of the embedding

	out             GRU
	out2            GRU
	keyOutbedding   *Node // (N x hiddenSize), where N is the number of keys known
	keyOutbedding_b *Node // (N) vector
	// chOutbedding    *Node // (N x hiddenSize), where N is the number of channels known
	// chOutbedding_b  *Node // (N) vector
	durOutbedding   *Node // (N x hiddenSize)
	durOutbedding_b *Node // (N) vector

	// corpuses.
	keyLookup map[byte]int
	durLookup map[uint]int
	keys      []byte
	durations []uint

	g *ExprGraph
}

// NewS2S creates a new Seq2Seq network. Input size is the size of the embedding. Hidden size is the size of the hidden layer
func NewS2S(hiddenSize, embSize int, keys []byte, durations []uint) *seq2seq {
	g := NewGraph()

	keySize := len(keys) + 2
	durationSize := len(durations) + 2

	keyLookup := make(map[byte]int)
	for i, k := range keys {
		keyLookup[k] = i
	}
	durLookup := make(map[uint]int)
	for i, d := range durations {
		durLookup[d] = i
	}

	dummyPrev := NewVector(g, Float, WithShape(hiddenSize), WithName("Dummy Prev"), WithInit(Zeroes()))
	dummyPrev2 := NewVector(g, Float, WithShape(hiddenSize), WithName("Dummy Prev2"), WithInit(Zeroes()))
	keyEmbedding := NewMatrix(g, Float, WithShape(keySize, embSize), WithName("Key Embedding"), WithInit(GlorotN(1.0)))
	durEmbedding := NewMatrix(g, Float, WithShape(durationSize, embSize), WithName("Duration Embedding"), WithInit(GlorotN(1.0)))

	// the reason for 3xembSize:
	// each entry (key, dur) has embsize
	// furthermore, there is an interaction variable (which is the hadamard prod of both entries).
	in := MakeGRU("In", g, 2*embSize, hiddenSize, Float)
	in2 := MakeGRU("In2", g, hiddenSize, hiddenSize, Float)
	out := MakeGRU("Out", g, 2*embSize, hiddenSize, Float)
	out2 := MakeGRU("Out2", g, hiddenSize, hiddenSize, Float)

	keyOutbedding := NewMatrix(g, Float, WithShape(keySize, hiddenSize), WithName("Key Outbedding"), WithInit(GlorotN(1.0)))
	keyOutbedding_b := NewVector(g, Float, WithShape(keySize), WithName("KeyOut bias"), WithInit(Zeroes()))
	durOutbedding := NewMatrix(g, Float, WithShape(durationSize, hiddenSize), WithName("Duration Outbedding"), WithInit(GlorotN(1.0)))
	durOutbedding_b := NewVector(g, Float, WithShape(durationSize), WithName("DurOut bias"), WithInit(Zeroes()))

	return &seq2seq{
		in:           in,
		in2:          in2,
		dummyPrev:    dummyPrev,
		dummyPrev2:   dummyPrev2,
		keyEmbedding: keyEmbedding,
		durEmbedding: durEmbedding,

		out:             out,
		out2:            out2,
		keyOutbedding:   keyOutbedding,
		keyOutbedding_b: keyOutbedding_b,
		durOutbedding:   durOutbedding,
		durOutbedding_b: durOutbedding_b,

		keyLookup: keyLookup,
		durLookup: durLookup,
		keys:      keys,
		durations: durations,

		g: g,
	}
}

func (s *seq2seq) learnables() []ValueGrad {
	retVal := make([]ValueGrad, 0)
	retVal = append(retVal, s.in.learnables()...)
	retVal = append(retVal, s.in2.learnables()...)
	retVal = append(retVal, s.out.learnables()...)
	retVal = append(retVal, s.out2.learnables()...)
	retVal = append(retVal, s.keyEmbedding, s.keyOutbedding, s.keyOutbedding_b)
	retVal = append(retVal, s.durEmbedding, s.durOutbedding, s.durOutbedding_b)
	return retVal
}

// train trains a pair of input and outputs
func (s *seq2seq) train(in []message, out []message) (cost *Node, err error) {
	var prev, prev2 *Node = s.dummyPrev, s.dummyPrev2
	for i := -1; i <= len(in); i++ {
		var keyIn, durIn int
		if i == -1 {
			keyIn, durIn = 0, 0
		} else if i == len(in) {
			keyIn, durIn = 1, 1
		} else {
			keyIn = s.keyLookup[in[i].key] + 2
			durIn = s.durLookup[in[i].duration] + 2
		}
		// log.Printf("KeyIn %v ChIn %v, durIn %v | %v", keyIn, chIn, durIn, s.keyEmbedding.Shape())

		keyVec := Must(Slice(s.keyEmbedding, S(keyIn)))
		durVec := Must(Slice(s.durEmbedding, S(durIn)))
		// interaction := Must(HadamardProd(keyVec, durVec))
		combined := Must(Concat(0, keyVec, durVec))
		if prev, err = s.in.Activate(combined, prev); err != nil {
			return
		}

		if prev2, err = s.in2.Activate(prev, prev2); err != nil {
			return
		}
	}

	// syllabus learning
	for i := -1; i < len(out); i++ {
		var keyIn, durIn int
		if i == -1 {
			keyIn, durIn = 0, 0
		} else {
			keyIn = s.keyLookup[out[i].key] + 2
			durIn = s.durLookup[out[i].duration] + 2
		}
		// log.Printf("Out KeyIn %v ChIn %v, durIn %v", keyIn, chIn, durIn)

		var targetKey, targetDur int
		if i == len(out)-1 {
			targetKey, targetDur = 1, 1
		} else {
			targetKey = s.keyLookup[out[i+1].key] + 2
			targetDur = s.durLookup[out[i+1].duration] + 2
		}

		// log.Printf("TargetKey %v, TargetCh %v, targetDur %v", targetKey, targetCh, targetDur)

		keyVec := Must(Slice(s.keyEmbedding, S(keyIn)))
		durVec := Must(Slice(s.durEmbedding, S(durIn)))
		// interaction := Must(HadamardProd(keyVec, durVec))
		combined := Must(Concat(0, keyVec, durVec))
		combined = Must(Rectify(combined))

		if prev, err = s.out.Activate(combined, prev); err != nil {
			return
		}
		if prev2, err = s.out2.Activate(prev, prev2); err != nil {
			return
		}
		predKey := Must(SoftMax(Must(Add(Must(Mul(s.keyOutbedding, prev2)), s.keyOutbedding_b))))
		predDur := Must(SoftMax(Must(Add(Must(Mul(s.durOutbedding, prev2)), s.durOutbedding_b))))

		// NLL
		keyProb := Must(Neg(Must(Log(predKey))))
		durProb := Must(Neg(Must(Log(predDur))))

		// loss
		keyLoss := Must(Slice(keyProb, S(targetKey)))
		durLoss := Must(Slice(durProb, S(targetDur)))

		if cost == nil {
			cost = Must(Add(keyLoss, durLoss))
		} else {
			cost = Must(Add(cost, Must(Add(keyLoss, durLoss))))
		}
	}
	return cost, nil

}

func (s *seq2seq) predict(in []message) (output []message, err error) {
	var prev, prev2 *Node = s.dummyPrev, s.dummyPrev2
	for i := -1; i <= len(in); i++ {
		var keyIn, durIn int
		if i == -1 {
			keyIn, durIn = 0, 0
		} else if i == len(in) {
			keyIn, durIn = 1, 1
		} else {
			var minKey int = int((^uint(0)) >> 1)
			for j, key := range s.keys {
				diff := int(key) - int(in[i].key)
				sq := diff * diff
				if sq < minKey {
					minKey = sq
					keyIn = j
				}
			}

			var minDur int = int((^uint(0)) >> 1)
			for j, dur := range s.durations {
				diff := int(dur) - int(in[i].duration)
				sq := diff * diff
				if sq < minDur {
					minDur = sq
					durIn = j
				}
			}
			log.Printf("Key %v Duration %v. Closest %v", in[i].key, in[i].duration, s.durations[durIn])
			keyIn += 2
			durIn += 2
		}

		keyVec := Must(Slice(s.keyEmbedding, S(keyIn)))
		durVec := Must(Slice(s.durEmbedding, S(durIn)))
		// interaction := Must(HadamardProd(keyVec, durVec))
		combined := Must(Concat(0, keyVec, durVec))
		if prev, err = s.in.Activate(combined, prev); err != nil {
			return
		}
		if prev2, err = s.in2.Activate(prev, prev2); err != nil {
			return
		}
	}

	var keyIn, durIn int
	for {
		keyVec := Must(Slice(s.keyEmbedding, S(keyIn)))
		durVec := Must(Slice(s.durEmbedding, S(durIn)))
		// interaction := Must(HadamardProd(keyVec, durVec))
		combined := Must(Concat(0, keyVec, durVec))
		combined = Must(Rectify(combined))

		if prev, err = s.out.Activate(combined, prev); err != nil {
			return
		}
		if prev2, err = s.out2.Activate(prev, prev2); err != nil {
			return
		}
		predKey := Must(SoftMax(Must(Add(Must(Mul(s.keyOutbedding, prev2)), s.keyOutbedding_b))))
		predDur := Must(SoftMax(Must(Add(Must(Mul(s.durOutbedding, prev2)), s.durOutbedding_b))))

		g := s.g.SubgraphRoots(predKey, predDur)
		machine := NewLispMachine(g, ExecuteFwdOnly())
		if err = machine.RunAll(); err != nil {
			log.Printf("FAIL WHILE PREDICTING %d", len(output))
			return
		}

		keyID := sample(predKey.Value())
		durID := sample(predDur.Value())

		// end
		if keyID <= 1 || durID < 2 {
			break
		}

		msg := message{
			channel:  1,
			key:      s.keys[keyID-2],
			duration: s.durations[durID-2],
		}

		output = append(output, msg)
		// count rests
		var restCount int
		for _, o := range output {
			if o.key == 255 {
				restCount++
			}
		}

		if len(output) >= maxOut {
			break
		}
		keyIn = s.keyLookup[msg.key]
		durIn = s.durLookup[msg.duration]
	}
	s.g.UnbindAllNonInputs()
	// log.Printf("ALL NODES %d", len(s.g.AllNodes()))
	// for _, n := range s.g.AllNodes() {
	// 	log.Printf("\n%v: %t %t", n, n.IsVar(), n.Value() == nil)
	// }
	// s.g.UnbindAllNonInputs()
	// log.Printf("ALL NODES2 %d", len(s.g.AllNodes()))
	// for _, n := range s.g.AllNodes() {
	// 	log.Printf("\n%v: %t %t", n, n.IsVar(), n.Value() == nil)
	// }
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

func train(s *seq2seq, iter int, solver Solver, data []trainingPair) (err error) {
	shuffle(data)

	var avgCost float32
	for i, pair := range data {
		var g *ExprGraph
		var cost *Node
		var costVal Value
		cost, err = s.train(pair.in, pair.out)
		read := Read(cost, &costVal)
		g = s.g.SubgraphRoots(read)

		// logger := log.New(os.Stderr, "", 0)
		// m := NewLispMachine(g, WithLogger(logger), LogBothDir(), WithWatchlist())
		m := NewLispMachine(g)
		if err = m.RunAll(); err != nil {
			if ctxError, ok := err.(contextualError); ok {
				log.Printf("FAIL WHILE TRAINING")
				log.Printf("Input %v", pair.in)
				log.Printf("Output %v", pair.out)
				log.Printf("ERR %+v", ctxError.Err())
			}
			return
			// ioutil.WriteFile("FAIL.dot", []byte(s.g.ToDot()), 0644)
			// return
		}
		avgCost += costVal.Data().(float32)
		if math32.IsNaN(avgCost) {
			s.g.UnbindAllNonInputs()
			return io.EOF
		}

		if iter == 0 || i == len(data)-1 {
			avgCost /= float32(len(data))
			log.Printf("Iter %d. Cost %v", iter, avgCost)
		}

		if err = solver.Step(s.learnables()); err != nil {
			return
		}
		if iter%100 == 0 && iter > 0 {
			if err = s.checkpoint(); err != nil {
				return
			}
		}
	}
	return nil

}
