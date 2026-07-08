(ns bjs-drones.telemetry
  "Synthetic UAV telemetry generator.
   Models realistic flight data + labeled anomaly injection."
  (:require [fastmath.random :as r]
            [fastmath.core :as m]))

;; --- Flight state ---

(defn initial-state []
  {:t 0
   :lat 48.1351
   :lon 11.5820
   :alt 100.0
   :vx 0.0 :vy 0.0 :vz 0.0
   :roll 0.0 :pitch 0.0 :yaw 0.0
   :battery 100.0
   :gps-fix 3
   :anomaly nil})

;; --- Normal flight dynamics ---

(defn step-normal [state dt rng]
  (let [noise #(r/frandom rng (- %) %)
        {:keys [lat lon alt vx vy vz yaw battery]} state
        ;; gentle maneuver
        ax (+ (noise 0.3) (* -0.05 vx))
        ay (+ (noise 0.3) (* -0.05 vy))
        az (+ (noise 0.1) (* -0.05 vz))
        vx' (+ vx (* ax dt))
        vy' (+ vy (* ay dt))
        vz' (+ vz (* az dt))
        ;; convert m/s to degrees (approx)
        dlat (* vy' dt 9.0e-6)
        dlon (* vx' dt 1.3e-5)
        dalt (* vz' dt)]
    (assoc state
           :t (+ (:t state) dt)
           :lat (+ lat dlat)
           :lon (+ lon dlon)
           :alt (max 0.0 (+ alt dalt))
           :vx vx' :vy vy' :vz vz'
           :roll (* (noise 2.0) 1.0)
           :pitch (* (noise 2.0) 1.0)
           :yaw (mod (+ yaw (* (noise 5.0) dt)) 360.0)
           :battery (max 0.0 (- battery (* 0.01 dt)))
           :gps-fix 3
           :anomaly nil)))

;; --- Anomaly injectors ---

(defn inject-gps-jam [state rng]
  (let [noise #(r/frandom rng (- %) %)]
    (assoc state
           :lat (+ (:lat state) (noise 0.005))
           :lon (+ (:lon state) (noise 0.005))
           :gps-fix 1
           :anomaly :gps-jam)))

(defn inject-motor-failure [state rng]
  (let [noise #(r/frandom rng (- %) %)]
    (assoc state
           :roll (+ (:roll state) (noise 30.0))
           :pitch (+ (:pitch state) (noise 30.0))
           :vz (- (:vz state) 2.0)
           :anomaly :motor-failure)))

(defn inject-battery-drain [state _rng]
  (assoc state
         :battery (max 0.0 (- (:battery state) 2.0))
         :anomaly :battery-drain))

(defn inject-hijack [state rng]
  (assoc state
         :yaw (r/frandom rng 0.0 360.0)
         :vx (* (:vx state) -1.5)
         :vy (* (:vy state) -1.5)
         :anomaly :hijack))

(def anomaly-fns
  {:gps-jam inject-gps-jam
   :motor-failure inject-motor-failure
   :battery-drain inject-battery-drain
   :hijack inject-hijack})

;; --- Flight simulation ---

(defn simulate-flight
  "Returns seq of telemetry records.
   anomaly-windows: [{:type kw :start int :end int}]"
  [{:keys [steps dt seed anomaly-windows]
    :or {steps 600 dt 0.5 seed 42 anomaly-windows []}}]
  (let [rng (r/rng :mersenne seed)
        anomaly-active? (fn [t]
                          (some (fn [{:keys [type start end]}]
                                  (when (<= start t end) type))
                                anomaly-windows))]
    (loop [state (initial-state)
           acc []]
      (if (>= (:t state) (* steps dt))
        acc
        (let [step (int (/ (:t state) dt))
              anom (anomaly-active? step)
              state' (step-normal state dt rng)
              state'' (if anom
                        ((anomaly-fns anom) state' rng)
                        state')]
          (recur state'' (conj acc state'')))))))

;; --- Multi-flight dataset ---

(defn generate-dataset
  "n-normal + one flight per anomaly type."
  [& {:keys [n-normal steps]
      :or {n-normal 5 steps 600}}]
  (let [normal-flights
        (for [i (range n-normal)]
          {:flight-id (str "normal-" i)
           :records (simulate-flight {:steps steps :seed i})})
        anomaly-flights
        (for [[kw _] anomaly-fns
              :let [start (int (* steps 0.3))
                    end   (int (* steps 0.6))]]
          {:flight-id (str "anomaly-" (name kw))
           :records (simulate-flight
                     {:steps steps
                      :seed (hash kw)
                      :anomaly-windows [{:type kw :start start :end end}]})})]
    (concat normal-flights anomaly-flights)))
