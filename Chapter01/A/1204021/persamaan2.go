package main

import (
	"fmt"
	"log"

	//  "io/ioutil"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// kode program fungsi z = Wx dimana W adalah matriks n kali n. x adalah vektor ukuran n. dengan n = 2.1957
func main() {
	g := G.NewGraph()

	var z *G.Node
	var err error

	//deklarasi W, dengan bobot inisiasi matB
	matB := []float64{0.8, 0.6, 0.5, 0.3}
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

	// fungsi z=Wx menggunakan rumus multification
	if z, err = G.Mul(mat, vec); err != nil {
		log.Fatal(err)
	}

	machine := G.NewTapeMachine(g)

	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	// ioutil.WriteFile("pers2_graph.dot", []byte(g.ToDot()), 0644)

	fmt.Println(z.Value().Data())
}
