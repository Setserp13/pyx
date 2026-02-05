
class Material:
  def __init__(self, name, albedo=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.5, albedo_texture=None, metallic_roughness_texture=None, normal_texture=None, ao_texture=None):
    self.name = name
    self.albedo = albedo
    self.metallic = metallic
    self.roughness = roughness
    self.albedo_texture = albedo_texture
    self.metallic_roughness_texture = metallic_roughness_texture
    self.normal_texture = normal_texture
    self.ao_texture = ao_texture
