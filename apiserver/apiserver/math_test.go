package apiserver

import (
	"math"
	"testing"
)

const eps = 1e-8

func equaleps(a, b float64) bool {
	return math.Abs(a-b) < eps
}

func TestFrobeniusNorm(t *testing.T) {
	norm := FrobeniusNorm([]float64{-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0})
	if !equaleps(norm, 7.745966692414834) {
		t.Errorf("Norm was incorrect, got: %f, want: %f.", norm, 7.745966692414834)
	}
}

func TestCosinMetric(t *testing.T) {
	norm := CosinMetric([]float64{0.46727048, 0.43004233, 0.27952332, 0.1524828, 0.47310451},
		[]float64{0.03538705, 0.81665373, 0.15395064, 0.29546334, 0.50521321})
	if !equaleps(norm, 0.8004287073454146) {
		t.Errorf("CosineMetric was incorrect, got: %f, want: %f.", norm, 0.8004287073454146)
	}
}
