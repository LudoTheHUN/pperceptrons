(ns pperceptrons.core-test
  (:require [clojure.test :refer :all]
            [pperceptrons.core :refer :all]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as m-ops]
            [clojure.core.matrix.implementations :as imp]))


(defn cover-2d-range [pp]
   (map  (fn [[x y]] (read-out pp [x y]))
         (apply concat (map (fn [x] (map (fn [y] [x y]) (range -1 1 0.1)) ) (range -1 1 0.1)))))


(defn frequencies-of-resonable-pp-2d [epsilon zerod?]
    (sort (frequencies (apply concat (map (fn [seed] (cover-2d-range (make-resonable-pp 2 epsilon zerod? seed))) (range 20))))))


(defn test-resonable-pp-responses [epsilon]
  (let [freqs (frequencies-of-resonable-pp-2d epsilon true)]
    (is (= (count freqs)  (int (+ (/ 1 epsilon) 1))))
    (is (= (apply max (map first freqs)) 1.0))
    (is (= (apply min (map first freqs)) -1.0))
    ))


#_(apply max (map first (frequencies-of-resonable-pp-2d 0.1 true)))

#_(deftest good-epsilon-values
 "Here we tests if that resonably created pp's cover the response range from -1 to +1 with the right number of intevals given some epsilon for centered pp's"
 (test-resonable-pp-responses 0.5)
 (test-resonable-pp-responses 0.25)
 (test-resonable-pp-responses 0.1)
 #_(test-resonable-pp-responses 0.05)    ;;This fails only becuase we do no try sufficent number of different seeded pp's
  )


#_(frequencies-of-resonable-pp-2d 0.10 false)
#_(frequencies-of-resonable-pp-2d 0.501 false)
#_(frequencies-of-resonable-pp-2d 0.10 true)
#_(frequencies-of-resonable-pp-2d 0.5 true)



;;1D pp learning tests

;;binary-pp example

(def data-1d-binary-fn-data [
    [[-1.0] -1.0] [[-0.9]  1.0] [[-0.8] -1.0]
    [[-0.5]  1.0] [[ 0.0] -1.0] [[ 0.5]  1.0]
    [[ 0.8] -1.0] [[ 0.9] -1.0] [[ 1.0] -1.0]
   ])

(def data-2d-binary-fn-data [
    [[-1.0 -1.0] -1.0] [[-0.9  0.5]  1.0] [[-0.8  0.2 ] -1.0]
    [[-0.5 -0.4]  1.0] [[ 0.0  1.0] -1.0] [[ 0.5 -0.1 ]  1.0]
    [[ 0.8 -0.8] -1.0] [[ 0.9  0.7] -1.0] [[ 1.0 -0.7 ] -1.0]
   ])


(def data-2d-3way-fn-data [
    [[-1.0 -1.0] -1.0] [[-0.9  0.5]  1.0] [[-0.8  0.2 ]  0.0]
    [[-0.5 -0.4]  1.0] [[ 0.0  1.0]  0.0] [[ 0.5 -0.1 ]  1.0]
    [[ 0.8 -0.8]  0.0] [[ 0.9  0.7] -1.0] [[ 1.0 -0.7 ] -1.0]
   ])



(defn multi-read-out [pp input-output-seq]
  (map (fn [test-data] (read-out pp (first test-data))) input-output-seq))

(defn number-of-correct-answers [pp input-output-seq]
   (map (fn [test-data] (= (read-out pp (first test-data)) (second test-data))) input-output-seq))

(defn true-fraction [true-false-seq]
  (/  (count (filter true? true-false-seq))
      (count true-false-seq)))

(defn test-trainging [pp input-output-seq n-epochs]
  ;trains and assesses a pp
  (let [pp-nepochs (train-seq-epochs pp input-output-seq n-epochs)]
{:pp pp-nepochs
 :correctness (true-fraction (number-of-correct-answers pp-nepochs input-output-seq))
 :epochs n-epochs}))

#_(test-trainging (make-resonable-pp 1 0.501 false 2 7) data-1d-binary-fn-data  100)
#_(test-trainging (make-resonable-pp 1 0.501 false 11 5) data-1d-binary-fn-data  100)

;;Testing over many seeds
(deftest good-epsilon-values "testing training over many seeds"
  ;;Note that the size-boost is used
    (is (=
         (sort (frequencies (pmap (fn [x] (:correctness (test-trainging (make-resonable-pp 1 0.501 false x 3)   ;;use boost to get more correct results if the input has more features
                                                          data-1d-binary-fn-data  300)           ;;epochs
                                                         )) (range 4)  )))   ;;how many seeds to try
          '([1 4])))

    (is (=
         (sort (frequencies (pmap (fn [x] (:correctness (test-trainging (make-resonable-pp 2 0.501 false x 2)   ;;use boost to get more correct results if the input has more features
                                                          data-2d-binary-fn-data  300)           ;;epochs
                                                         )) (range 4)  )))   ;;how many seeds to try
          '([1 4])))

    (is (=
         (sort (frequencies (pmap (fn [x] (:correctness (test-trainging (make-resonable-pp 2 0.5 true x 2)   ;;use boost to get more correct results if the input has more features
                                                          data-2d-3way-fn-data  300)           ;;epochs
                                                         )) (range 4)  )))   ;;how many seeds to try
          '([1 4])))
 )


;;This almost needs a sub project for parameter optimisation




(defn make-test-data [])


(apply concat (repeat 3 data-1d-binary-fn-data))

(range -1 1 0.1)








