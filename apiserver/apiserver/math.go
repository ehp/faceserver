package apiserver

import (
	"math"
)

// replaced with single run version of CosinMetric
func FrobeniusNorm(arr []float64) float64 {
	sum := 0.0
	for _, num := range arr {
		anum := math.Abs(num)
		sum += anum * anum
	}
	return math.Sqrt(sum)
}

func CosinMetric(x []float64, y []float64) float64 {
	l := len(x)
	if l != len(y) {
		return -1.0
	}

	xsum := 0.0
	ysum := 0.0
	sum := 0.0
	idx := 0
	for idx < l {
		xabs := math.Abs(x[idx])
		yabs := math.Abs(y[idx])
		xsum += xabs * xabs
		ysum += yabs * yabs
		sum += x[idx] * y[idx]
		idx++
	}

	return sum / (math.Sqrt(xsum) * math.Sqrt(ysum))
}
