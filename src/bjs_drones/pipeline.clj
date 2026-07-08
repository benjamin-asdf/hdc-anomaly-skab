(ns bjs-drones.pipeline
  "Feature extraction and windowing over telemetry records."
  (:require [tablecloth.api :as tc]
            [fastmath.stats :as stats]))

;; --- Records -> Dataset ---

(defn records->ds [flight-id records]
  (-> (tc/dataset records)
      (tc/add-column :flight-id (repeat flight-id))
      (tc/select-columns [:flight-id :t :lat :lon :alt
                          :vx :vy :vz :roll :pitch :yaw
                          :battery :gps-fix :anomaly])))

(defn flights->ds [flights]
  (->> flights
       (map (fn [{:keys [flight-id records]}]
              (records->ds flight-id records)))
       (apply tc/concat)))

;; --- Derived features ---

(defn add-derived [ds]
  (-> ds
      (tc/add-column :speed
                     (fn [row]
                       (let [vx (get row :vx)
                             vy (get row :vy)
                             vz (get row :vz)]
                         (map #(Math/sqrt (+ (* %1 %1) (* %2 %2) (* %3 %3)))
                              vx vy vz))))
      (tc/add-column :attitude-magnitude
                     (fn [row]
                       (let [r (get row :roll)
                             p (get row :pitch)]
                         (map #(Math/sqrt (+ (* %1 %1) (* %2 %2))) r p))))))

;; --- Sliding window features ---

(defn window-stats
  "Given a seq of numbers, return {:mean :std :min :max}."
  [xs]
  (let [v (double-array xs)]
    {:mean (stats/mean v)
     :std  (stats/stddev v)
     :min  (reduce min xs)
     :max  (reduce max xs)}))

(defn sliding-windows
  "Partition records into overlapping windows, extract features per window."
  [records {:keys [window-size step]
            :or {window-size 20 step 10}}]
  (let [feature-cols [:speed :attitude-magnitude :alt :battery :gps-fix]]
    (->> records
         (partition window-size step)
         (map (fn [window]
                (let [anomaly-label (or (some :anomaly window) :normal)]
                  (merge
                   {:window-start (:t (first window))
                    :window-end   (:t (last window))
                    :anomaly-label anomaly-label}
                   (into {}
                         (for [col feature-cols
                               :let [vals (keep col window)]
                               :when (seq vals)
                               [k v] (window-stats vals)]
                           [(keyword (str (name col) "-" (name k))) v])))))))))

(defn extract-features
  "Full pipeline: flights -> windowed feature matrix."
  [flights & {:keys [window-size step]
              :or {window-size 20 step 10}}]
  (let [ds (-> flights flights->ds add-derived)]
    (->> (tc/group-by ds :flight-id)
         :data
         (mapcat (fn [group-ds]
                   (let [flight-id (first (get group-ds :flight-id))
                         records   (tc/rows group-ds :as-maps)]
                     (->> (sliding-windows records {:window-size window-size
                                                    :step step})
                          (map #(assoc % :flight-id flight-id)))))))))
