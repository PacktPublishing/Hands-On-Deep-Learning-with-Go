package main

import (
	"fmt"
	"log"
	"math"

	"gorgonia.org/gorgonia"
)

func main() {
	g := gorgonia.NewGraph()

	// definisi inputan variable x
	x := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))

	// Definisi variable a,b,c
	a := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("a"))
	b := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("b"))
	c := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("c"))

	// Definisi aturan Persamaan Trigonometri
	term1 := gorgonia.Must(gorgonia.Sin(x))
	term2 := gorgonia.Must(gorgonia.Cos(x))

	// Definisi Persamaan
	equation := gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Mul(a, term1)), gorgonia.Must(gorgonia.Mul(b, term2))))

	// Bikin VM untuk proses hitung
	machine := gorgonia.NewTapeMachine(g)
	defer machine.Close()

	// Atur nilai input
	gorgonia.Let(x, math.Pi/4)
	gorgonia.Let(a, 2.0)
	gorgonia.Let(b, 3.0)
	gorgonia.Let(c, 4.0)

	// Hitung Trigonometri
	if err := machine.RunAll(); err != nil {
		log.Fatal(err)
	}
	result := equation.Value().Data().(float64)

	fmt.Printf("Hasil Hitung Trigonometri: %f\n", result)
}
