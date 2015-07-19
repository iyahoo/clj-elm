(ns clj-elm.data
  (:require [clojure.repl :refer [doc]]
            [incanter.datasets :as da]
            [incanter.core :as c]
            [incanter.stats :as st]
            [incanter.io :as io]))

(def australian
  (io/read-dataset "data/australian.csv" :delim \, :header true))

(defn num-of-feature 
  ([dataset]
   {:pre [(or (c/matrix? dataset) (c/dataset? dataset))]}
   (dec (c/ncol dataset))))

(defn num-of-data 
  ([dataset]
   {:pre [(or (c/matrix? dataset) (c/dataset? dataset))]}
   (c/nrow dataset)))

(defn get-features 
  ([line]
   {:pre [(coll? line)]}
   (butlast line)))

(defn class-label 
  ([dataset]
   {:pre [(or (c/matrix? dataset) (c/dataset? dataset))]}
   (map int (c/to-vect (c/$ :all 14 dataset)))))

(defn ith-feature-list 
  ([dataset i]
   {:pre [(or (c/matrix? dataset) (c/dataset? dataset))]}
   (c/$ i dataset)))

(defn each-ith-f 
  ([f dataset n]
   {:pre [(or (c/matrix? dataset) (c/dataset? dataset))]}
   (let [fnum (num-of-feature dataset)
         clabel (class-label dataset)]
     (->> (range 0 fnum)
          (map #(f (ith-feature-list dataset %)))
          (vec)
          (#(conj % n))))))

(defn each-ith-mean 
  ([dataset]
   (each-ith-f st/mean dataset 0)))

(defn each-ith-sd 
  ([dataset]
   (each-ith-f st/sd dataset 1)))

(defn normalize 
  ([dataset]
   {:pre [(or (c/matrix? dataset) (c/dataset? dataset))]
    :post [(= (count %))]}
   (let [means (each-ith-mean dataset)
         sds (each-ith-sd dataset)]    
     (->> (c/to-vect dataset)
          (map #(map - % means))
          (map #(map / % sds))
          (c/to-dataset)))))
