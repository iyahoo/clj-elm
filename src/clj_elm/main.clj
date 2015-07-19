(ns clj-elm.main
  (:require [clojure.repl :refer [doc]]
            [clj-elm.core :as core]
            [clj-elm.data :as data]
            [incanter.io :as io]
            [incanter.core :as c]
            [incanter.stats :as st]))

(def australian
  (io/read-dataset "data/australian.csv" :delim \, :header true))

(defn -main [dataset]
  (let [d (data/num-of-feature dataset)
        L 20
        ass (core/make-ass d L)
        bs (core/make-bs L)
        xss (map data/get-features (c/to-vect (data/normalize dataset)))
        H (core/hidden-layer-output-matrix ass bs xss)
        T (data/class-label dataset)
        betas (c/to-vect (c/mmult (core/pseudo-inverse-matrix H) T))]
    (map #(core/output betas ass bs %) xss)))
