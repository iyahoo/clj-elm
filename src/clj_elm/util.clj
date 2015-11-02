(ns clj-elm.util)

(defn printfl [obj]
  {:post [(coll? %) (nil? (first %))]}
  [(print obj) (flush)])
