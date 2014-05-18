(ns pperceptrons.performance-tests
  (:require [clojure.test :refer :all]
            [pperceptrons.core :refer :all]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as m-ops]
            [clojure.core.matrix.implementations :as imp]
            [criterium.core :as crit]))



(def do-benchmarks? false)




(if do-benchmarks?
 (crit/quick-bench  (map m/length (m/slices (scale-to-size-one (uniform-dist-matrix-center-0 :vectorz [50 1000] 42))))))  ;15.741735 ms
