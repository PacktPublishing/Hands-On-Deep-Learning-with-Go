package main

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

func main() {
	g := gorgonia.NewGraph()

	// definisi konstanta-konstanta
	a := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("a"))
	b := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("b"))
	c := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("c"))

	// definisi variabel x
	x := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))

	// persamaan parabola
	y := gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Mul(a, gorgonia.Must(gorgonia.Pow(x, gorgonia.NewConstant(2.0))))), gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Mul(b, x)), c))))

	// kompilasi dan eksekusi graph
	machine := gorgonia.NewTapeMachine(g)
	defer machine.Close()

	// assign nilai konstanta-konstanta
	gorgonia.Let(a, 2.0)
	gorgonia.Let(b, 3.0)
	gorgonia.Let(c, 4.0)

	// assign nilai variabel x
	gorgonia.Let(x, 5.0)

	// jalankan perhitungan
	if err := machine.RunAll(); err != nil {
		fmt.Println(err)
	}

	// tampilkan hasil
	fmt.Printf("y = %v", y.Value())
}
