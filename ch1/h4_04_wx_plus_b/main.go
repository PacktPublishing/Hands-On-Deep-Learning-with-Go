package main

import (
	"fmt"
	"log"
	"io/ioutil"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	g := G.NewGraph()

	matB := []float64{0.9,0.7,0.4,0.2}
	matT := tensor.New(tensor.WithBacking(matB), tensor.WithShape(2, 2))
	mat := G.NewMatrix(g,
		   tensor.Float64,
		   G.WithName("W"),
		   G.WithShape(2, 2),
		   G.WithValue(matT),
	)
	
	vecB := []float64{5,7}
	vecT := tensor.New(tensor.WithBacking(vecB), tensor.WithShape(2))

	vec := G.NewVector(g,
		tensor.Float64,    
		G.WithName("x"),  
		G.WithShape(2),    
		G.WithValue(vecT), 
	)

	b := G.NewScalar(g, tensor.Float64, G.WithName("b"), G.WithValue(3.0))

	z, err := G.Add(G.Must(G.Mul(mat, vec)), b)

	// create a VM to run the program on
	machine := G.NewTapeMachine(g)

	// set initial values then run

	if machine.RunAll() != nil {
		log.Fatal(err)
	}

	fmt.Println(z.Value().Data())
	// Output: [12.399999999999999 6.4]

	ioutil.WriteFile("simple_graph.dot", []byte(g.ToDot()), 0644)
}