package main

import (
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"runtime"

	pb "gopkg.in/cheggaaa/pb.v1"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

const (
	embeddingSize = 20
	maxOut        = 11

	// gradient update stuff
	l2reg     = 0.000001
	learnrate = 0.01
	clipVal   = 5.0
)

var trainiter = flag.Int("iter", 0, "How many iterations to train")
var toCondition = flag.Bool("condition", false, "Condition the NN to #2?")

// var trainingData = flag.String("train", "simplediag.mid", "What is the MIDI file to use for training? Channel 0 is the input,  Channel 1 and above are responses")

// various global variable inits
var epochSize = -1
var inputSize = -1
var outputSize = -1

const corpus string = "the cat sat on the mat"

var dt tensor.Dtype = tensor.Float32

// type trainingPair string
type message struct {
	channel  byte
	key      byte
	duration uint
	velocity byte // velocity 0 == noteoff
}

type trainingPair struct {
	in, out []message
}

// WHERE TO 1HV THE INPUITS?????
// func OneHotVector(id, classes int, t tensor.Dtype, opts ...NodeConsOpt) *Node {

func trainingLoop(s2s *seq2seq, iters int, pairs []trainingPair, embeddingSize, hiddenSize int, keys []byte, durations []uint, mOut string) {
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithLearnRate(learnrate), gorgonia.WithL2Reg(l2reg), gorgonia.WithClip(clipVal))
	bar := pb.StartNew(iters)

	for i := 0; i < iters; i++ {
		ioutil.WriteFile("tmp.dot", []byte(s2s.g.ToDot()), 0644)
		fmt.Println("iter: ", i)
		if err := train(s2s, i, solver, pairs[:]); err != nil && err != io.EOF {
			log.Fatalf("Training Failure %+v", err)
		}
		bar.Increment()
		if i%100 == 0 && i > 0 {
			s2s.g.UnbindAll()

			s2s = NewS2S(embeddingSize, hiddenSize, keys, durations)
			runtime.GC() // reduce memory pressure
			if err := s2s.load(); err != nil {
				log.Fatalf("GC Pressure Reduction Failure", err)
			}
		}
	}
	if iters > 50 {
		if err := s2s.checkpoint(); err != nil {
			log.Fatalf("Failed to save checkpoint after training", err)
		}
	}
	bar.Finish()

	// notify user that the neural network is ready
	// mOut.WriteShort(0x90, int64(keys[0]), 100)
	// mOut.WriteShort(0x90, int64(keys[len(keys)-1]), 100)
	// time.Sleep(1 * time.Second)
	// mOut.WriteShort(0x80, int64(keys[0]), 0)
	// mOut.WriteShort(0x80, int64(keys[len(keys)-1]), 0)
}

func main() {
	flag.Parse()

	var (
		pairs     []trainingPair
		keys      []byte
		durations []uint
	)

	hiddenSize := 100

	s2s := NewS2S(embeddingSize, hiddenSize, keys, durations)

	// try to load
	var mOut string
	var iters = *trainiter
	// if err := s2s.load(); err != nil {
	// 	log.Printf("Loading failed %v", err)
	// 	iters = 10000
	// }
	// initVocab(sentences, 1)
	fmt.Printf("vocab typeval: %T%v\nvocabIndex typeval: %T%v\n", vocab, vocab, vocabIndex, vocabIndex)
	trainingLoop(s2s, iters, pairs, embeddingSize, hiddenSize, keys, durations, mOut)
	// go MIDILoop(mIn, mOut, s2s)
	// mainGL()

}
