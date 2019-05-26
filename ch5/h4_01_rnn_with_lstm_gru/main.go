package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"runtime/pprof"

	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
var memprofile = flag.String("memprofile", "", "write memory profile to this file")

// TODO/Questions:

// Input:
// extend vocab.go to import txt
//  - got ur dict got ur vocab got ur vocabindex
//  - replace sentencesRaw
//  - slice into chars. map to []rune
//  - produce paired input/output examples (input, input + 1 across corpus)
//  - 1HV the lot? sparsity meh?

// Model:
// - why keySize = len of keys +2?
// - definition of ValueGrad?

//  Main:
//  - keys/durations/mOut?
//  - why pointer for trainiter?

const (
	embeddingSize = 20
	maxOut        = 30

	// gradient update stuff
	l2reg     = 0.000001
	learnrate = 0.01
	clipVal   = 5.0
)

var trainiter = flag.Int("iter", 5, "How many iterations to train")
var inputFile = flag.String("filename", "shakespeare.txt", "Filename of text to train on")

// various global variable inits
var epochSize = -1
var inputSize = -1
var outputSize = -1

// const corpus string = "shakespeare.txt"
// const corpus string = `the cat sat on the mat
// hello world
// wild stalyns

var corpus string

func init() {
	buf, err := ioutil.ReadFile("shakespeare.txt")
	if err != nil {
		panic(err)
	}
	corpus = string(buf)
}

var dt tensor.Dtype = tensor.Float32

// type trainingPair string

// type pair struct {
// 	t string
// 	tplusone string
// }

// type trainingPair struct {
// 	in, out []message
// }

// WHERE TO 1HV THE INPUITS?????
// func OneHotVector(id, classes int, t tensor.Dtype, opts ...NodeConsOpt) *Node {

func main() {
	flag.Parse()
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	// f, err := os.Create("trace.out")
	// if err != nil {
	// 	panic(err)
	// }
	// defer f.Close()

	// err = trace.Start(f)
	// if err != nil {
	// 	panic(err)
	// }
	// defer trace.Stop()

	hiddenSize := 100

	s2s := NewS2S(hiddenSize, embeddingSize, vocab)
	solver := NewRMSPropSolver(WithLearnRate(learnrate), WithL2Reg(l2reg), WithClip(clipVal), WithBatchSize(float64(len(sentences))))
	for k, v := range vocabIndex {
		log.Printf("%q %v", k, v)
	}

	p, h, w, err := Heatmap(s2s.decoder.Value().(*tensor.Dense))
	p.Save(w, h, "embn0.png")

	if err := train(s2s, 300, solver, sentences); err != nil {
		panic(err)
	}
	out, err := s2s.predict([]rune(corpus))
	if err != nil {
		panic(err)
	}
	fmt.Printf("OUT %q\n", out)

	p, h, w, err = Heatmap(s2s.decoder.Value().(*tensor.Dense))
	p.Save(w, h, "embn.png")
}
