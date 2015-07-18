(ns clj-elm.data
  (:require [clojure.repl :refer [doc]]
            [incanter.datasets :as data]
            [incanter.core :as c]
            [incanter.stats :as st]))

(defn num-of-feature [dataset]
  (dec (c/ncol dataset)))

(defn get-class-label [dataset]
  (map int (c/$ :col0 dataset)))

(defn get-features [row]
  (rest row))

;; (defn normalize [dataset]
;;   (let [;; v-data (c/to-vect dataset)
;;         xs (map #(c/$ % dataset) (range 1 (c/ncol dataset)))
;;         means (cons 0 (map #(st/mean %) xs))
;;         sds (cons 1 (map #(st/sd %) xs))]
;;     (c/to-vect (c/matrix-map #(c/div (c/minus % means) sds) dataset))))

(defn normalize [dataset]
  (let [v-data (c/to-vect dataset)
        xs (map #(c/$ % dataset) (range 1 (c/ncol dataset)))
        means (cons 0 (map #(st/mean %) xs))
        sds (cons 1 (map #(st/sd %) xs))]
    (map #(c/to-vect (c/div (c/minus % means) sds)) v-data)))
