package main

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

func main() {
	g := gorgonia.NewGraph()

	var x, y, z *gorgonia.Node

	// define the expression
	x = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	y = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))
	z, _ = gorgonia.Add(x, y)

	// create a VM to run the program on
	machine := gorgonia.NewTapeMachine(g)

	// set initial values then run
	gorgonia.Let(x, 2.0)
	gorgonia.Let(y, 2.5)
	_ = machine.RunAll()
	fmt.Printf("%v", z.Value())

	defer machine.Close()

}
