package main

import (
	"fmt"
	"io/ioutil"
	"log"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	g := G.NewGraph()

	var z *G.Node
	var err error

	// define the expression
	//deklarasi W, dengan bobot inisiasi matB
	matB := []float64{0.9, 0.7, 0.4, 0.2}
	matT := tensor.New(tensor.WithBacking(matB), tensor.WithShape(2, 2))
	mat := G.NewMatrix(g, tensor.Float64, G.WithName("W"), G.WithShape(2, 2), G.WithValue(matT))

	// deklarasi x dengan inisiasi bobot vecB
	vecB := []float64{5, 7}

	vecT := tensor.New(tensor.WithBacking(vecB), tensor.WithShape(2))

	vec := G.NewVector(g,
		tensor.Float64,
		G.WithName("x"),
		G.WithShape(2),
		G.WithValue(vecT),
	)
	if z, err = G.Mul(mat, vec); err != nil {
		log.Fatal(err)
	}

	// create a VM to run the program on
	machine := G.NewTapeMachine(g)
	defer machine.Close()

	machine.RunAll()
	//melihat hasil output
	fmt.Println(z.Value().Data())

	ioutil.WriteFile("simple_graph.dot", []byte(g.ToDot()), 0644)
}
