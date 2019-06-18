package main

type Memory struct {
	State        Point
	Action       Vector
	Reward       float32
	NextState    Point
	NextMovables []Vector
	isDone       bool
}
