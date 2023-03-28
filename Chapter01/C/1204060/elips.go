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

	// definisi variabel x dan y
	x := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	y := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))

	// persamaan elips
	eq1 := gorgonia.Must(gorgonia.Pow(x, gorgonia.NewConstant(2.0)))
	eq2 := gorgonia.Must(gorgonia.Pow(y, gorgonia.NewConstant(2.0)))
	eq3 := gorgonia.Must(gorgonia.Pow(a, gorgonia.NewConstant(2.0)))
	eq4 := gorgonia.Must(gorgonia.Pow(b, gorgonia.NewConstant(2.0)))
	eq5 := gorgonia.Must(gorgonia.Div(eq1, eq3))
	eq6 := gorgonia.Must(gorgonia.Div(eq2, eq4))
	eq7 := gorgonia.Must(gorgonia.Add(eq5, eq6))
	eq8 := gorgonia.Must(gorgonia.Sub(gorgonia.NewConstant(1.0), eq7))

	// kompilasi dan eksekusi graph
	machine := gorgonia.NewTapeMachine(g)
	defer machine.Close()

	// assign nilai konstanta-konstanta
	gorgonia.Let(a, 2.0)
	gorgonia.Let(b, 3.0)

	// assign nilai variabel x dan y
	gorgonia.Let(x, 1.0)
	gorgonia.Let(y, 2.0)

	// jalankan perhitungan
	if err := machine.RunAll(); err != nil {
		fmt.Println(err)
	}

	// tampilkan hasil
	fmt.Printf("nilai persamaan elips: %v", eq8.Value())
}
