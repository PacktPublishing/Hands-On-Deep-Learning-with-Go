package main

import (
	"fmt"
	"io/ioutil"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Deklarasikan fungsi main, dan inisiasi NewGraph() untuk deklarasi membuat graph komputasi
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

	a, _ := G.Mul(mat, vec)
	z, _ := G.Add(a, b)

	//Buat VM object agar bisa menjalankan model fungsi g yang dideklarasikan pada langkah 2.
	machine := G.NewTapeMachine(g)

	machine.RunAll()
	//melihat hasil output
	fmt.Println(z.Value().Data())

	ioutil.WriteFile("simple_graph3.dot", []byte(g.ToDot()), 0644)
}
