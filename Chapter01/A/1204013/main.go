package main

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main1() {
	// Membuat graph baru
	g := gorgonia.NewGraph()

	// Membuat node input x
	x := gorgonia.NewTensor(g, tensor.Float64, 1, gorgonia.WithName("x"), gorgonia.WithShape(4), gorgonia.WithInit(gorgonia.Gaussian(0, 1)))

	// Menghitung nilai sin(x)
	sinX, err := gorgonia.Sin(x)
	if err != nil {
		log.Fatal(err)
	}

	// Membuat VM baru
	machine := gorgonia.NewTapeMachine(g)

	// Menjalankan VM
	if err := machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	// Mengambil hasil
	result := sinX.Value().Data().([]float64)

	// Mencetak hasil
	fmt.Println(result)

}
