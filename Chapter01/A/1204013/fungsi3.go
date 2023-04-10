package main

import (
	"fmt"
	"log"

	//  "io/ioutil"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// kode program fungsi z = Wx + b
func main() {
	g := G.NewGraph()

	//deklarasi W, dengan bobot inisiasi matB
	matB := []float64{0.9, 0.7, 0.4, 0.2}
	matT := tensor.New(tensor.WithBacking(matB), tensor.WithShape(2, 2))
	mat := G.NewMatrix(g,
		tensor.Float64,
		G.WithName("W"),
		G.WithShape(2, 2),
		G.WithValue(matT),
	)

	// deklarasi x dengan inisiasi bobot vecB
	vecB := []float64{5, 7}

	vecT := tensor.New(tensor.WithBacking(vecB), tensor.WithShape(2))

	vec := G.NewVector(g,
		tensor.Float64,
		G.WithName("x"),
		G.WithShape(2),
		G.WithValue(vecT),
	)

	//tambah deklarasi b
	b := G.NewScalar(g,
		tensor.Float64,
		G.WithName("b"),
		G.WithValue(3.0),
	)

	a, err := G.Mul(mat, vec)
	z, err := G.Add(a, b)

	if a, err = G.Mul(mat, vec); err != nil {
		log.Fatal(err)
	}

	if z, err = G.Add(a, b); err != nil {
		log.Fatal(err)
	}

	machine := G.NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	// ioutil.WriteFile("pers3_graph.dot", []byte(g.ToDot()), 0644)

	fmt.Println(z.Value().Data())
}
