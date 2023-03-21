package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"github.com/aiteung/mnist"
	"github.com/pkg/errors"
	"gopkg.in/cheggaaa/pb.v1"
	"image"
	"image/jpeg"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	epochs     = flag.Int("epochs", 5, "Number of epochs to train for")
	dataset    = flag.String("dataset", "train", "Which dataset to train on? Valid options are \"train\" or \"test\"")
	dtype      = flag.String("dtype", "float64", "Which dtype to use")
	batchsize  = flag.Int("batchsize", 100, "Batch size")
	cpuprofile = flag.String("cpuprofile", "", "CPU profiling")
)

const loc = "./mnist/"
const backup = "./backup/"

var dt tensor.Dtype

func parseDtype() {
	switch *dtype {
	case "float64":
		dt = tensor.Float64
	case "float32":
		dt = tensor.Float32
	default:
		log.Fatalf("Unknown dtype: %v", *dtype)
	}
}

type nn struct {
	g              *gorgonia.ExprGraph
	w0, w1, w2, w3 *gorgonia.Node

	out     *gorgonia.Node
	predVal gorgonia.Value
}

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }

func newNN(g *gorgonia.ExprGraph) *nn {
	// Create node for w/weight

	weight0, weight1, weight2, weight3, err := readFromBackup(backup + "backup1.gob")
	if err != nil {
		panic(err)
	}

	w0 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(784, 128), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	w1 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(128, 64), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	w2 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(64, 128), gorgonia.WithName("w2"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	w3 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(128, 784), gorgonia.WithName("w3"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))

	gorgonia.Let(w0, weight0)
	gorgonia.Let(w1, weight1)
	gorgonia.Let(w2, weight2)
	gorgonia.Let(w3, weight3)

	return &nn{
		g:  g,
		w0: w0,
		w1: w1,
		w2: w2,
		w3: w3,
	}
}

func (m *nn) learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0, m.w1, m.w2, m.w3}
}

func (m *nn) fwd(x *gorgonia.Node) (err error) {
	var l0, l1, l2, l3, l4 *gorgonia.Node
	var l0dot, l1dot, l2dot, l3dot *gorgonia.Node

	// Set first layer to be copy of input
	l0 = x

	// Dot product of l0 and w0, use as input for Sigmoid
	if l0dot, err = gorgonia.Mul(l0, m.w0); err != nil {
		return errors.Wrap(err, "Unable to multiple l0 and w0")
	}
	l1 = gorgonia.Must(gorgonia.Sigmoid(l0dot))

	if l1dot, err = gorgonia.Mul(l1, m.w1); err != nil {
		return errors.Wrap(err, "Unable to multiple l1 and w1")
	}
	l2 = gorgonia.Must(gorgonia.Sigmoid(l1dot))

	if l2dot, err = gorgonia.Mul(l2, m.w2); err != nil {
		return errors.Wrap(err, "Unable to multiple l2 and w2")
	}
	l3 = gorgonia.Must(gorgonia.Sigmoid(l2dot))

	if l3dot, err = gorgonia.Mul(l3, m.w3); err != nil {
		return errors.Wrap(err, "Unable to multiple l3 and w3")
	}
	l4 = gorgonia.Must(gorgonia.Sigmoid(l3dot))

	// m.pred = l3dot
	// gorgonia.Read(m.pred, &m.predVal)
	// return nil

	m.out = l4
	gorgonia.Read(l4, &m.predVal)
	return

}

const pixelRange = 255

func reversePixelWeight(px float64) byte {
	// return byte((pixelRange*px - pixelRange) / 0.9)
	return byte(pixelRange*math.Min(0.99, math.Max(0.01, px)) - pixelRange)
}

func visualizeRow(x []float64) *image.Gray {
	// since this is a square, we can take advantage of that
	l := len(x)
	side := int(math.Sqrt(float64(l)))
	r := image.Rect(0, 0, side, side)
	img := image.NewGray(r)

	pix := make([]byte, l)
	for i, px := range x {
		pix[i] = reversePixelWeight(px)
	}
	img.Pix = pix

	return img
}

func main() {
	flag.Parse()
	parseDtype()
	rand.Seed(7945)

	// // intercept Ctrl+C
	// sigChan := make(chan os.Signal, 1)
	// signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	// doneChan := make(chan bool, 1)

	var inputs tensor.Tensor
	var err error

	// load our data set
	trainOn := *dataset
	if inputs, _, err = mnist.Load(trainOn, loc, dt); err != nil {
		log.Fatal(err)
	}

	numExamples := inputs.Shape()[0]
	bs := *batchsize

	// MNIST data consists of 28 by 28 black and white images
	// however we've imported it directly now as 784 different pixels
	// as a result, we need to reshape it to match what we actually want
	// if err := inputs.Reshape(numExamples, 1, 28, 28); err != nil {
	// 	log.Fatal(err)
	// }

	// we should now also proceed to put in our desired variables
	// x is where our input should go, while y is the desired output
	g := gorgonia.NewGraph()
	// x := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(bs, 1, 28, 28), gorgonia.WithName("x"))
	x := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 784), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 784), gorgonia.WithName("y"))

	// Init variables
	m := newNN(g)

	if err = m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	losses, err := gorgonia.Square(gorgonia.Must(gorgonia.Sub(y, m.out)))
	if err != nil {
		log.Fatal(err)
	}
	cost := gorgonia.Must(gorgonia.Mean(losses))
	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)

	// cost = gorgonia.Must(gorgonia.Neg(cost))
	// create a VM to run the program on
	machine := gorgonia.NewTapeMachine(g)
	defer machine.Close()

	if inputs, _, err = mnist.Load("test", loc, dt); err != nil {
		log.Fatal(err)
	}

	numExamples = inputs.Shape()[0]
	bs = *batchsize

	batches := numExamples / bs
	log.Printf("Batches %d", batches)
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Second / 20)
	bar.SetMaxWidth(80)

	batches = numExamples / bs

	bar = pb.New(batches)
	bar.SetRefreshRate(time.Second / 20)
	bar.SetMaxWidth(80)
	bar.Prefix(fmt.Sprintf("Epoch Test"))
	bar.Set(0)
	bar.Start()
	for b := 0; b < batches; b++ {
		start := b * bs
		end := start + bs
		if start >= numExamples {
			break
		}
		if end > numExamples {
			end = numExamples
		}

		var xVal tensor.Tensor
		if xVal, err = inputs.Slice(sli{start, end}); err != nil {
			log.Fatal("Unable to slice x")
		}

		// if yVal, err = inputs.Slice(sli{start, end}); err != nil {
		// 	log.Fatal("Unable to slice y")
		// }
		// if err = xVal.(*tensor.Dense).Reshape(bs, 1, 28, 28); err != nil {
		// 	log.Fatal("Unable to reshape %v", err)
		// }
		if err = xVal.(*tensor.Dense).Reshape(bs, 784); err != nil {
			log.Fatal("Unable to reshape %v", err)
		}

		gorgonia.Let(x, xVal)
		gorgonia.Let(y, xVal)
		if err = machine.RunAll(); err != nil {
			log.Fatalf("Failed at epoch test: %v", err)
		}

		for j := 0; j < xVal.Shape()[0]; j++ {
			rowT, _ := xVal.Slice(sli{j, j + 1})
			row := rowT.Data().([]float64)

			img := visualizeRow(row)

			f, _ := os.OpenFile(fmt.Sprintf("images/%d - %d input.jpg", b, j), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
			jpeg.Encode(f, img, &jpeg.Options{jpeg.DefaultQuality})
			f.Close()
		}

		arrayOutput := m.predVal.Data().([]float64)
		yOutput := tensor.New(tensor.WithShape(bs, 784), tensor.WithBacking(arrayOutput))

		for j := 0; j < yOutput.Shape()[0]; j++ {
			rowT, _ := yOutput.Slice(sli{j, j + 1})
			row := rowT.Data().([]float64)

			img := visualizeRow(row)

			f, err := os.OpenFile(fmt.Sprintf("images/%d - %d output.jpg", b, j), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
			if err != nil {
				fmt.Printf("\nError terjadi : %v \n", err)
			}

			jpeg.Encode(f, img, &jpeg.Options{jpeg.DefaultQuality})
			f.Close()
		}

		machine.Reset()
		bar.Increment()
	}
	log.Printf("Epoch Test | cost %v", costVal)
	bar.Finish()
}

func readFromBackup(file string) (w0 tensor.Dense, w1 tensor.Dense, w2 tensor.Dense, w3 tensor.Dense, err error) {
	f, err := os.Open(file)
	if err != nil {
		return
	}
	defer f.Close()
	dec := gob.NewDecoder(f)
	log.Println("decoding xT")
	err = dec.Decode(&w0)
	if err != nil {
		return
	}
	log.Println("decoding yT")

	err = dec.Decode(&w1)
	if err != nil {
		return
	}
	err = dec.Decode(&w2)
	if err != nil {
		return
	}
	err = dec.Decode(&w3)
	if err != nil {
		return
	}
	return
}
