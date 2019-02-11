package cifar

import (
	"io/ioutil"
	"log"
	"math"
	"os"
	"path/filepath"

	"gorgonia.org/tensor"
)

const numLabels = 10
const pixelRange = 255

func pixelWeight(px byte) float64 {
	retVal := float64(px)/pixelRange*0.9 + 0.1
	if retVal == 1.0 {
		return 0.999
	}
	return retVal
}

func reversePixelWeight(px float64) byte {
	// return byte((pixelRange*px - pixelRange) / 0.9)
	return byte(pixelRange*math.Min(0.99, math.Max(0.01, px)) - pixelRange)
}

// Load function for cifar
// typ can be "train" or "test"
// loc should be where the CIFAR-10 files can be found
func Load(typ, loc string) (inputs, targets tensor.Tensor, err error) {
	// cifar-10 comes in 6 separate binary files

	var arrayFiles []string
	switch typ {
	case "train":
		arrayFiles = []string{
			"data_batch_1.bin",
			"data_batch_2.bin",
			"data_batch_3.bin",
			"data_batch_4.bin",
			"data_batch_5.bin",
		}
	case "test":
		arrayFiles = []string{
			"test_batch.bin",
		}
	}

	// create slices to store our data
	var labelSlice []uint8
	var imageSlice []float64

	// each binary file comes formatted in 3073 byte groups
	// 1 byte for the class
	// 32 by 32 bytes for each of the red, green and blue pixel colour values
	for _, targetFile := range arrayFiles {
		f, err := os.Open(filepath.Join(loc, targetFile))
		if err != nil {
			log.Fatal(err)
		}

		defer f.Close()
		cifar, err := ioutil.ReadAll(f)

		if err != nil {
			log.Fatal(err)
		}

		for index, element := range cifar {
			if index%3073 == 0 {
				labelSlice = append(labelSlice, uint8(element))
			} else {
				imageSlice = append(imageSlice, pixelWeight(element))
			}
		}
	}

	// transform label slice into the necessary format
	labelBacking := make([]float64, len(labelSlice)*numLabels, len(labelSlice)*numLabels)
	labelBacking = labelBacking[:0]
	for i := 0; i < len(labelSlice); i++ {
		for j := 0; j < numLabels; j++ {
			if j == int(labelSlice[i]) {
				labelBacking = append(labelBacking, 0.9)
			} else {
				labelBacking = append(labelBacking, 0.1)
			}
		}
	}

	inputs = tensor.New(tensor.WithShape(len(labelSlice), 3, 32, 32), tensor.WithBacking(imageSlice))
	targets = tensor.New(tensor.WithShape(len(labelSlice), numLabels), tensor.WithBacking(labelBacking))
	return
}
