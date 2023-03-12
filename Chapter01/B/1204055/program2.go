package main
import (
     "fmt"
	 "io/ioutil"
     . "gorgonia.org/gorgonia"
     "gorgonia.org/tensor"
)

func main() {
	g := NewGraph()
  //deklarasi W, dengan bobot inisiasi matB
matB := []float64{0.9,0.7,0.4,0.2}
matT := tensor.New(tensor.WithBacking(matB), tensor.WithShape(2, 2))
mat := NewMatrix(g,
        tensor.Float64,
        WithName("W"),
        WithShape(2, 2),
        WithValue(matT),
)

// deklarasi x dengan inisiasi bobot vecB
vecB := []float64{5,7}

vecT := tensor.New(tensor.WithBacking(vecB), tensor.WithShape(2))

vec := NewVector(g,
        tensor.Float64,
        WithName("x"),
        WithShape(2),
        WithValue(vecT),
)

z, _ := Mul(mat, vec)
machine := NewTapeMachine(g)
machine.RunAll()
//melihat hasil output
fmt.Println(z.Value().Data())
ioutil.WriteFile("simple_graph.dot", []byte(g.ToDot()), 0644)
}

