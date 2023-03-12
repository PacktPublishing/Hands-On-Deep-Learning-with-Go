package main

import (
	"fmt"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	g := G.NewGraph()

	//deklarasi W, dengan bobot inisiasi matB
matB := []float64{0.9,0.7,0.4,0.2}
matT := tensor.New(tensor.WithBacking(matB), tensor.WithShape(2, 2))
mat := G.NewMatrix(g,
        tensor.Float64,
        G.WithName("W"),
        G.WithShape(2, 2),
        G.WithValue(matT),
)

// deklarasi x dengan inisiasi bobot vecB
vecB := []float64{5,7}

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

machine := G.NewTapeMachine(g)

machine.RunAll()
//melihat hasil output
fmt.Println(z.Value().Data())
  }