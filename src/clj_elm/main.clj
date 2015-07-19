(ns clj-elm.main
  (:require [clojure.repl :refer [doc]]
            [clj-elm.core :refer [make-ass make-bs hidden-layer-output-matrix pseudo-inverse-matrix output]]
            [clj-elm.data :refer [num-of-feature get-features normalize class-label]]
            [incanter.io :as io]
            [incanter.core :refer [to-vect matrix mmult]]
            [incanter.stats :as st]))

(def australian
  (io/read-dataset "data/australian.csv" :delim \, :header true))

(defn -main [dataset]
  (let [d (num-of-feature dataset)
        L 20
        ass (make-ass d L)
        bs (make-bs L)
        xss (map get-features (to-vect (normalize dataset)))
        H (hidden-layer-output-matrix ass bs xss)
        T (class-label dataset)
        betas (to-vect (mmult (pseudo-inverse-matrix H) T))]
    (map #(output betas ass bs %) xss)))
