package main

import (
	"math/rand"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func sampleT(val gorgonia.Value) int {
	var t tensor.Tensor
	var ok bool
	if t, ok = val.(tensor.Tensor); !ok {
		panic("Expects a tensor")
	}

	return tensor.SampleIndex(t)
}

func sample(val gorgonia.Value) int {

	var t tensor.Tensor
	var ok bool
	if t, ok = val.(tensor.Tensor); !ok {
		panic("expects a tensor")
	}
	indT, err := tensor.Argmax(t, -1)
	if err != nil {
		panic(err)
	}
	if !indT.IsScalar() {
		panic("Expected scalar index")
	}
	return indT.ScalarValue().(int)
}

func shuffle(a []string) {
	for i := len(a) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		a[i], a[j] = a[j], a[i]
	}
}

type byteslice []byte

func (s byteslice) Len() int           { return len(s) }
func (s byteslice) Less(i, j int) bool { return s[i] < s[j] }
func (s byteslice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

type uintslice []uint

func (s uintslice) Len() int           { return len(s) }
func (s uintslice) Less(i, j int) bool { return s[i] < s[j] }
func (s uintslice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
