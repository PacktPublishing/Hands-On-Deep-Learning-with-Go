package main

import (
	"fmt"
	"io/ioutil"
	"log"

	. "gorgonia.org/gorgonia"
)

func main() {
	g := NewGraph()

	var a, b, c *Node
	var err error

	a = NewScalar(g, Float64, WithName("a"))
	b = NewScalar(g, Float64, WithName("b"))
	if c, err = Add(a, b); err != nil {
		log.Fatal(err)
	}

	machine := NewTapeMachine(g)

	Let(a, 1.0)
	Let(b, 2.0)
	machine.RunAll()

	fmt.Printf("%v", c.Value())

	ioutil.WriteFile("simple_graph1.dot", []byte(g.ToDot()), 0644)
}
