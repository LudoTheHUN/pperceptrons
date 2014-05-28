(ns pperceptrons.performance-tests
  (:require [clojure.test :refer :all]
            [pperceptrons.core :refer :all]
            [pperceptrons.core-test :refer :all]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as m-ops]
            [clojure.core.matrix.implementations :as imp]
            [criterium.core :as crit]))



(def do-benchmarks? false)




(if do-benchmarks?
 (do
  (crit/quick-bench  (map m/length (m/slices (scale-to-size-one (uniform-dist-matrix-center-0 :vectorz [50 1000] 42)))))



    ;15.741735 ms



(time (def pp-perf (:pp (test-trainging (make-resonable-pp 2 0.12 false :seed 42 :size-boost 10)   ;;use boost to get more correct results if the input has more features
                                                   some-analytical-fn-data 200))))

(crit/quick-bench
(read-out pp-perf [-0.5 0.5])
 )

  ;68.459495 µs



)
)








;;Example from README.md


(def input [
    [[-1.0  1.0] -1.0] [[ 1.0  1.0]  1.0]
    [[-1.0 -1.0]  1.0] [[ 1.0 -1.0] -1.0]
   ])
(def ppa
 (make-resonable-pp
   2   ;inputsize    ;;how wide is the input
   0.501   ;epsilon      ;How accurate do you need to be. Use 0.501 for a binary pperceptron (which will return 1.0 or 1.0). Smaller epsilon will make the pp bigger internally.
   false   ;zerod?       ;true here make the number of perceptrons even, so it will be possible to respond with 0.0 as the anser
      ; & ops
  ; :seed 42
   ))

(train ppa (ffirst input) (second (first input)))


(def pp-trained  (let [n-epochs 600]
                    (train-seq-epochs ppa input n-epochs)))

(read-out pp-trained [-1.0  1.0])   ;=> -1.0
(read-out pp-trained [ 1.0  1.0])   ;=>  1.0
(read-out pp-trained [-1.0 -1.0])   ;=>  1.0
(read-out pp-trained [ 1.0 -1.0])   ;=> -1.0


(read-out ppa [-1.0  1.0])   ;=>  1.0
(read-out ppa [ 1.0  1.0])   ;=> -1.0
(read-out ppa [-1.0 -1.0])   ;=> -1.0
(read-out ppa [ 1.0 -1.0])   ;=>  1.0

(read-out pp-trained [-1.8  1.7])  ;=> -1.0
(read-out pp-trained [ 1.8  1.7])  ;=>  1.0
(read-out pp-trained [-1.8 -1.7])  ;=>  1.0
(read-out pp-trained [ 1.8 -1.7])  ;=> -1.0

(read-out pp-trained [-0.8  0.7])  ;=> -1.0
(read-out pp-trained [ 0.8  0.7])  ;=>  1.0
(read-out pp-trained [-0.8 -0.7])  ;=>  1.0
(read-out pp-trained [ 0.8 -0.7])  ;=> -1.0

(read-out pp-trained [-0.9  0.9])  ;=> -1.0
(read-out pp-trained [ 0.9  0.9])  ;=>  1.0
(read-out pp-trained [-0.9 -0.9])  ;=>  1.0
(read-out pp-trained [ 0.9 -0.9])  ;=> -1.0

(read-out pp-trained [-1.1  1.1])   ;=> -1.0
(read-out pp-trained [ 1.1  1.1])   ;=>  1.0
(read-out pp-trained [-1.1 -1.1])   ;=>  1.0
(read-out pp-trained [ 1.1 -1.1])   ;=> -1.0

(read-out pp-trained [-1.05  1.05])   ;=> -1.0
(read-out pp-trained [ 1.05  1.05])   ;=>  1.0
(read-out pp-trained [-1.05 -1.05])   ;=>  1.0
(read-out pp-trained [ 1.05 -1.05])   ;=> -1.0




(read-out pp-trained [ 0.0 -0.0])
