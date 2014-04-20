(ns pperceptrons.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as m-ops]))


;;Resources
;https://github.com/mikera/core.matrix
;https://github.com/mikera/core.matrix/blob/master/src/main/clojure/clojure/core/matrix/examples.clj
;https://github.com/clojure-numerics/core.matrix.stats/blob/develop/src/main/clojure/clojure/core/matrix/random.clj


;http://www.igi.tugraz.at/psfiles/pdelta-journal.pdf

(def a (m/array [-1 2 -0.1]))

(def b (m/array [0.1 0.3 -0.4]))

(class a)
(m/shape a)

(m/mmul a b)


(defn input->z--input-vector [input-vector]
     (m/array (conj input-vector -1.0))
  )

(class -1.0)
(input->z--input-vector [1.0 2.0])




(defn perceptron-f [a--perceptron-weight-vector z--input-vector]
     (let [mmulresult (m/mmul a--perceptron-weight-vector z--input-vector)]
       (if (pos? mmulresult) 1.0 -1.0))
  )

;; we will need to re-implement the above in the learning steps so that we don't compute the same values more then once

(perceptron-f a b)


;pperceptron is made of many a--perceptron-weight-vectors

(def pperceptron
  ;;a pp with 3 perceptrons, taken in 3dim input, -1 in last dim for the threshold.
  (m/matrix
   [[-4.0 0.3 -0.3 0.5]
   [0.1 -0.2 -0.4 -0.6]
   [0.2 -0.1 -0.7 -0.6]]))

(class pperceptron)

(defn total-pperceptron [pperceptron input-vector]
  (let [perceptron_value_fn (fn [perceptron] (perceptron-f perceptron input-vector))]
    (reduce +
            (map perceptron_value_fn  (m/slices pperceptron)))
  ))


(total-pperceptron pperceptron  [-0.2 -0.2 10.5 -1.0])
(total-pperceptron pperceptron  [-0.2 -0.2 -10.5 -1.0])
(total-pperceptron pperceptron  [-0.2 -0.2 1.5 -1.0])




(defn sp--squashing-function [pp-total rho--squashing-parameter]
     (cond
         (> pp-total rho--squashing-parameter)
           1.0
         (< pp-total (- rho--squashing-parameter))
           -1.0
         :else
         (/ pp-total rho--squashing-parameter)
   ))


(defn b-sp--binary-squashing-function [pp-total]
    (if (pos? pp-total ) 1.0 -1.0))



(defn pp-output [pperceptron input rho--squashing-parameter]
              (sp--squashing-function (total-pperceptron pperceptron    (input->z--input-vector input))   rho--squashing-parameter)
  )




(pp-output pperceptron [-0.2 0.2 0.3] 3)
(pp-output pperceptron [-10.2 0.2 0.6] 10)


(def epsilon 0.3)   ;;how accureate we want to be, must be > 0

(def rho--squashing-parameter 10.0)  ;; whole numbers, 1 means a binary pperceptron

(defn f-abs [n]
 (cond
   (neg? n) (- n)
   :else n))


(def eta--learning-rate 0.1)


(defn perceptron-f [a--perceptron-weight-vector z--input-vector]
     (let [mmulresult (m/mmul a--perceptron-weight-vector z--input-vector)]
       (if (pos? mmulresult) 1.0 -1.0))
  )


(defn pdelta-update [pperceptron  input target-output epsilon rho--squashing-parameter eta--learning-rate]
   (let [z--input-vector (input->z--input-vector input)

         perceptron_value_fn (fn [perceptron] (m/mmul perceptron z--input-vector))
         per-perceptron-totals  (map perceptron_value_fn  (m/slices pperceptron))
         out  (sp--squashing-function (reduce + (map #(if (pos? %) 1.0 -1.0) per-perceptron-totals)) rho--squashing-parameter)
         out-vs-train-abs (f-abs (- out target-output))
         ]

       (cond
         (<= out-vs-train-abs epsilon)   ;;pp gave correct enough answer, so nothing to do
            pperceptron
         (> out (+ target-output epsilon))   ;;pp is producing too higher result
         ; ;:pperceptron_lower
            (m/matrix (map  (fn [perceptron perceptron_value]
                               (if (pos? perceptron_value)
                                   (m-ops/- perceptron (mops/* z--input-vector eta--learning-rate))
                                   perceptron))
                            pperceptron
                            per-perceptron-totals))
         (< out (- target-output epsilon))   ;;pp is producing too lower result
         ; :pperceptron_higer
            (m/matrix (map  (fn [perceptron perceptron_value]
                               (if (neg? perceptron_value)
                                   (m-ops/+ perceptron (mops/* z--input-vector eta--learning-rate))
                                   perceptron))
                            pperceptron
                            per-perceptron-totals))

           :else
             pperceptron)
     ;;per-perceptron-totals

  ))


(mops/* input eta--learning-rate )

(def input [-1.1 -3 0.3])


(pdelta-update pperceptron input 0.2 0.01 10.0 0.1)




(def gamma--margin-around-zero 0.1)


(defn scaling-adela-fn [perceptron eta--learning-rate]
   (m-ops/* perceptron (* -1.0 eta--learning-rate (- (m/length-squared perceptron) 1.0)))
  )

(scaling-fn [0.2 0.0 0.4] 0.01)


;(defn perceptron-f-amount [a--perceptron-weight-vector z--input-vector]
;     (m/mmul a--perceptron-weight-vector z--input-vector))

;(defn perceptron-f-vote [mmulresult]
;     (if (pos? mmulresult) 1.0 -1.0))

(defn pdelta-update-with-margin [pperceptron  input target-output epsilon rho--squashing-parameter eta--learning-rate mu-zeromargin-importance gamma--margin-around-zero]
   (let [z--input-vector (input->z--input-vector input)

         perceptron_value_fn (fn [perceptron] (m/mmul perceptron z--input-vector))
         per-perceptron-totals  (map perceptron_value_fn  (m/slices pperceptron))
         out  (sp--squashing-function (reduce + (map #(if (pos? %) 1.0 -1.0) per-perceptron-totals)) rho--squashing-parameter)
         out-vs-train-abs (f-abs (- out target-output))
         ]

       (cond
         (<= out-vs-train-abs epsilon)   ;;pp gave correct enough answer, so nothing to do
             (m/matrix (map  (fn [perceptron perceptron_value]  (m-ops/+ perceptron (scaling-fn perceptron eta--learning-rate) )  )
                            pperceptron
                            per-perceptron-totals))
         (> out (+ target-output epsilon))   ;;pp is producing too higher result
         ; pperceptron must be lowered
            (m/matrix (map  (fn [perceptron perceptron_value]
                               (if (pos? perceptron_value)
                                   (m-ops/+ perceptron (scaling-fn perceptron eta--learning-rate) (m-ops/* z--input-vector -1.0 eta--learning-rate))
                                   (m-ops/+ perceptron (scaling-fn perceptron eta--learning-rate) ) ))
                            pperceptron
                            per-perceptron-totals))
         (< out (- target-output epsilon))   ;;pp is producing too lower result
         ; pperceptron must be made to produce higher results
            (m/matrix (map  (fn [perceptron perceptron_value]
                               (if (neg? perceptron_value)
                                   (m-ops/+ perceptron (scaling-fn perceptron eta--learning-rate) (m-ops/* z--input-vector       eta--learning-rate))
                                   (m-ops/+ perceptron (scaling-fn perceptron eta--learning-rate) ) ))
                            pperceptron
                            per-perceptron-totals))

         (<= out (+ target-output epsilon) )   ;;output a bit above target ... increase all perceptrons that are close to zero
             (m/matrix (map  (fn [perceptron perceptron_value]
                               (if (and (pos? perceptron_value) (< perceptron_value gamma--margin-around-zero) )
                                   (m-ops/+ perceptron (scaling-fn perceptron eta--learning-rate) (m-ops/* z--input-vector  mu-zeromargin-importance  eta--learning-rate))
                                   (m-ops/+ perceptron (scaling-fn perceptron eta--learning-rate) ) ))
                            pperceptron
                            per-perceptron-totals))
         (>= out (- target-output epsilon) )   ;;output a bit above target ... increase all perceptrons that are close to zero
             (m/matrix (map  (fn [perceptron perceptron_value]
                               (if (and (neg? perceptron_value) (< (* -1.0 gamma--margin-around-zero) perceptron_value ) )  ;;WIP factor out so its done once
                                   (m-ops/+ perceptron (scaling-fn perceptron eta--learning-rate) (m-ops/* z--input-vector -1.0  mu-zeromargin-importance  eta--learning-rate))
                                   (m-ops/+ perceptron (scaling-fn perceptron eta--learning-rate) ) ))
                            pperceptron
                            per-perceptron-totals))
        :else
                          (m/matrix (map  (fn [perceptron perceptron_value]  (m-ops/+ perceptron (scaling-fn perceptron eta--learning-rate) )  )
                            pperceptron
                            per-perceptron-totals)))
     ;;per-perceptron-totals
     ;out
  ))



pperceptron
input
(pdelta-update-with-margin pperceptron input 1.0
                           0.01 3.0 0.001 0.1 0.1)


(m/mmul [[0.1 0.2 0.3]
       [0.2 0.2 0.1]
       [0.3 0.1 -0.3]
       ]

      [1 2 3]
      )


(m/mmul [[0.1 0.2 0.3]
       [0.2 0.2 0.1]
       [0.3 0.1 -0.3]
       ]

      [[3 3 3]
       [2 2 2]
       [1 1 1]]
      )






