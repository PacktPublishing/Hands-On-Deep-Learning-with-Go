package main

import (
    "fmt"
    "log"

    "gorgonia.org/gorgonia"
    "gorgonia.org/tensor"
)

func main() {
    // Membuat graph baru
    g := gorgonia.NewGraph()

    // Membuat node input x
    x := gorgonia.NewTensor(g, tensor.Float64, 1, gorgonia.WithName("x"), gorgonia.WithShape(5), gorgonia.WithInit(gorgonia.Gaussian(0, 1)))

    // Membuat variabel-variabel untuk persamaan parabola y = ax^2 + bx + c
    a := gorgonia.NewScalar(g, tensor.Float64, gorgonia.WithName("a"), gorgonia.WithValue(0.5))
    b := gorgonia.NewScalar(g, tensor.Float64, gorgonia.WithName("b"), gorgonia.WithValue(1.0))
    c := gorgonia.NewScalar(g, tensor.Float64, gorgonia.WithName("c"), gorgonia.WithValue(2.0))

    // Membuat operasi-operasi untuk persamaan parabola
    xSquared := gorgonia.Must(gorgonia.Square(x))
    axSquared := gorgonia.Must(gorgonia.HadamardProd(xSquared, a))
    bx := gorgonia.Must(gorgonia.HadamardProd(x, b))
    yPred := gorgonia.Must(gorgonia.Add(axSquared, bx))
    yPred = gorgonia.Must(gorgonia.Add(yPred, c))

    // Membuat VM baru
    machine := gorgonia.NewTapeMachine(g)

    // Menjalankan VM
    if err := machine.RunAll(); err != nil {
        log.Fatal(err)
    }

    // Mengambil hasil
    result := yPred.Value().Data().([]float64)

    // Mencetak hasil
    fmt.Println(result)
}
