class Request:
    def __init__(self, origin, destination, request_time, latest_pickup, earliest_arrival, id, x_coord=None, y_coord=None, dest_x=None, dest_y=None):
        self.origin = origin
        self.destination = destination
        self.trr = request_time            # request time
        self.tplr = latest_pickup          # latest allowed pickup time
        self.t_star = earliest_arrival     # earliest possible dropoff
        self.id = str(id)
        self.x = x_coord # Store x_coord as 'x'
        self.y = y_coord # Store y_coord as 'y'
        self.dest_x = dest_x
        self.dest_y = dest_y

    def __repr__(self):
      return f"R{self.id}"

    # Add comparison methods to make Request objects orderable and hashable
    def __lt__(self, other):
        if not isinstance(other, Request):
            return NotImplemented
        return self.id < other.id

    def __eq__(self, other):
        if not isinstance(other, Request):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

class Vehicle:
    def __init__(self, start_node, start_time, onboard=None, id="vehicle"):
        self.qv = start_node
        self.tv = start_time
        self.Pv = onboard if onboard else []
        self.id = str(id)

    def __repr__(self):
        return f"V{self.id}"

def trip_name(trip_fset):
    req_ids = sorted([r.id for r in trip_fset])
    return "T[" + ",".join(req_ids) + "]"