package main

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
)

func main() {
	g := gorgonia.NewGraph()

	var x, y, z *gorgonia.Node
	var err error

	// define the expression
	x = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	y = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))
	if z, err = gorgonia.Add(x, y); err != nil {
		log.Fatal(err)
	}

	// create a VM to run the program on
	machine := gorgonia.NewTapeMachine(g)
	defer machine.Close()

	// set initial values then run
	gorgonia.Let(x, 2.0)
	gorgonia.Let(y, 2.5)
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%v", z.Value())
}

//hasilnya
// 4.5
