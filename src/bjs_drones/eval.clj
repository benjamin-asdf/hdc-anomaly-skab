(ns bjs-drones.eval
  "Train/test split, threshold search, and comparison table."
  (:require [bjs-drones.baseline :as bl]
            [bjs-drones.hdc :as hdc]))

(defn stratified-split
  "Split windows per class: train-frac goes to train, rest to test."
  [windows & {:keys [train-frac] :or {train-frac 0.7}}]
  (let [by-label (group-by :anomaly-label windows)]
    (reduce
     (fn [[train test] [_lbl ws]]
       (let [shuffled (shuffle ws)
             n-train  (max 1 (int (* train-frac (count shuffled))))
             ;; Always keep at least 1 in test
             n-train  (min n-train (dec (count shuffled)))]
         [(into train (take n-train shuffled))
          (into test  (drop n-train shuffled))]))
     [[] []]
     by-label)))

(defn tune-threshold
  "Grid search threshold on test-windows, optimise F1."
  [bl-model test-windows]
  (let [thresholds (range 1.0 8.0 0.25)
        best (apply max-key
                    :f1
                    (for [t thresholds]
                      (assoc (bl/evaluate bl-model test-windows :threshold t)
                             :threshold t)))]
    best))

(defn run-comparison
  "Full comparison: baseline vs HDC, returns result map."
  [all-windows & {:keys [train-frac] :or {train-frac 0.7}}]
  (let [[train-ws test-ws] (stratified-split all-windows :train-frac train-frac)
        normal-train (filter #(= (:anomaly-label %) :normal) train-ws)

        ;; Baseline
        bl-model   (bl/fit normal-train)
        bl-tuned   (tune-threshold bl-model test-ws)

        ;; HDC
        hdc-model  (hdc/train train-ws)
        hdc-result (hdc/evaluate hdc-model test-ws)]

    {:train-n     (count train-ws)
     :test-n      (count test-ws)
     :label-dist  (frequencies (map :anomaly-label all-windows))
     :baseline    bl-tuned
     :hdc         hdc-result
     :hdc-model   hdc-model
     :bl-model    bl-model
     :test-windows test-ws}))

(defn print-report [result]
  (println "\n=== Dataset ===")
  (println "Train:" (:train-n result) "| Test:" (:test-n result))
  (println "Label dist:" (:label-dist result))

  (println "\n=== Baseline (z-score one-class) ===")
  (let [bl (:baseline result)]
    (println (format "Threshold: %.2f" (double (:threshold bl))))
    (println (format "Accuracy:  %.3f" (double (:accuracy bl))))
    (println (format "Precision: %.3f" (double (:precision bl))))
    (println (format "Recall:    %.3f" (double (:recall bl))))
    (println (format "F1:        %.3f" (double (:f1 bl)))))

  (println "\n=== HDC Multi-class Classifier ===")
  (let [hdc (:hdc result)]
    (println (format "Accuracy: %.3f" (double (:accuracy hdc))))
    (println "Per-class:")
    (doseq [[lbl stats] (sort-by key (:per-class hdc))]
      (println (format "  %-20s  acc=%.2f  n=%d"
                       (name lbl)
                       (double (:accuracy stats))
                       (int (:n stats)))))))
