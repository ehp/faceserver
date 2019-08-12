// Copyright 2019 Petr Masopust, Aprar s.r.o.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
