class TrackableObject:
	def __init__(self, objectID, centroid, radio, frame_name, detected):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]
		self.radios = [radio]
		self.observations = [{'image_name': frame_name,
		 'x': centroid[0], 'y': centroid[1],
		  'r': radio, 'detection': detected}]

	def add_observation(self, centroid, radio, frame_name, detected):
		self.centroids.append(centroid)
		self.radios.append(radio)
		self.observations.append({'image_name': frame_name,
		 'x': centroid[0], 'y': centroid[1],
		  'r': radio, 'detection': detected})

