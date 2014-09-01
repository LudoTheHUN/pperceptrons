(ns pperceptrons.frugalize)

(defn make-frugal-estimator [seed fage]
  (let [rnd (java.util.Random. seed)
        estimate (atom 10.0)]
    (fn [data-point]
        (let [random (.nextDouble rnd)]
          (if (> random fage)
                (if (> data-point @estimate )
                        (swap! estimate (fn [x] (* x 1.1)))
                        (swap! estimate (fn [x] (* x 0.9))) )
             @estimate
            )))
  ))

(quote
 ;examples

;fastest moving estimator
(def estimator1 (make-frugal-estimator 0 0.0))

;slower moving estimator
(def estimator2 (make-frugal-estimator 0 0.9))


(estimator1 10)
(estimator1 10)
(estimator1 10)
(estimator1 10)
(estimator1 10)

(estimator2 10)
(estimator2 10)
(estimator2 10)
(estimator2 10)
(estimator2 10)


(estimator1 5)
(estimator1 1)
(estimator1 1)
(estimator1 1)
(estimator1 1)

(estimator2 5)
(estimator2 1)
(estimator2 1)
(estimator2 1)
(estimator2 1)


;short_term_error
;long_term_error

;short_term_error > long_term_error   ->> decrease learning rate
;short_term_error < long_term_error   ->> increase learning rate


)
