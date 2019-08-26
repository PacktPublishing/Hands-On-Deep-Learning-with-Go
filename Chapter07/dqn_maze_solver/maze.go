package main

import (
	"bytes"
	"log"
	"math/rand"
	"reflect"
	"strings"
	"time"

	mazegen "github.com/itchyny/maze"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
)

type tile byte

const (
	empty tile = iota
	wall
	start
	goal
	player
)

var tiletype tensor.Dtype

func init() {
	tiletype = tensor.Dtype{reflect.TypeOf(tile(1))}
}

type Point struct{ X, Y int }
type Vector Point

type Maze struct {
	// some maze object
	*mazegen.Maze
	repr   *tensor.Dense
	iter   [][]tile
	values [][]float32

	player, start, goal Point

	// meta

	r *rand.Rand
}

func NewMaze(h, w int) *Maze {
	m := mazegen.NewMaze(h, w)
	m.Generate()
	f := mazegen.Default

	var buf bytes.Buffer
	m.Print(&buf, f)

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	for i, line := range lines {
		lines[i] = strings.TrimSpace(line)
	}

	width := len(lines[0]) / 2
	height := len(lines)
	log.Printf("width %d, height %d", width, height)

	var flat []tile
	for y := 0; y < height; y++ {
		if y >= len(lines) {
			continue
		}
		for x := 0; x < width; x++ {
			if x*2 >= len(lines[y]) {
				continue
			}
			switch lines[y][x*2 : x*2+2] {
			case "##":
				flat = append(flat, wall)
			case "S ", " S", "S:", ":S":
				flat = append(flat, start)

			case "G ", " G", "G:", ":G":
				flat = append(flat, goal)
			default:
				flat = append(flat, empty)
			}
		}
	}
	repr := tensor.New(tensor.WithBacking(flat))
	repr.Reshape(2*h+1, 2*w+1)
	mat, _ := native.Matrix(repr)
	iter := mat.([][]tile)
	values := make([][]float32, len(iter))

	var startPoint, goalPoint Point
	r := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
	for i := range iter {
		values[i] = make([]float32, len(iter[i]))

		for j := range iter[i] {
			switch iter[i][j] {
			case wall:
				values[i][j] = -100
			case start:
				startPoint = Point{i, j}
				values[i][j] = -100
			case goal:
				goalPoint = Point{i, j}
				values[i][j] = 100
			default:
				rv := r.Intn(2)
				if rv == 1 {
					values[i][j] = -1
				} else {
					values[i][j] = 0
				}
			}

		}
	}

	return &Maze{
		Maze:   m,
		repr:   repr,
		iter:   iter,
		values: values,
		start:  startPoint, goal: goalPoint, player: startPoint,
		r: r} // ,
}

func (m *Maze) CanMoveTo(player Point, direction Vector) bool {
	dir := Point(direction)
	newX, newY := player.X+dir.X, player.Y+dir.Y
	if newX < 0 || newX >= len(m.iter) {
		return false
	}
	if newY < 0 || newY >= len(m.iter[0]) {
		return false
	}

	return m.iter[newY][newX] != wall
}

func (m *Maze) Move(direction Vector) {
	m.iter[m.player.Y][m.player.X] = empty

	m.player.X += direction.X
	m.player.Y += direction.Y

	m.iter[m.player.Y][m.player.X] = player
}

func (m *Maze) Value(action Vector) (float32, bool) {
	pos := Point{m.player.X + action.X, m.player.Y + action.Y}
	if pos.X < 0 || pos.X > len(m.values[0]) {
		return -100, false
	}
	if pos.Y < 0 || pos.Y > len(m.values) {
		return -100, false
	}
	if pos.X == m.goal.X && pos.Y == m.goal.Y {
		return m.values[pos.Y][pos.X], true
	}
	return m.values[pos.Y][pos.X], false
}

// Reset moves player back to start position
func (m *Maze) Reset() {
	m.iter[m.player.Y][m.player.X] = empty
	m.player = m.start
	m.iter[m.player.Y][m.player.X] = player
}
