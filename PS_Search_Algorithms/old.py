# def connected_only6(self, conf_space):
#     nn = []
#     xi, yi, thetai = self.ind()
#     x_size, y_size, thetai_size = conf_space.space.shape
#
#     nn.append(((xi+1) % x_size, yi, thetai))
#     nn.append(((xi-1) % x_size, yi, thetai))
#     nn.append((xi, (yi+1) % y_size, thetai))
#     nn.append((xi, (yi-1) % y_size, thetai))
#     nn.append((xi, yi, (thetai+1) % thetai_size))
#     nn.append((xi, yi, (thetai-1) % thetai_size))
#
#     # nn.remove(self.ind())
#     parent = self.parent
#     while parent is not None:
#         if parent.ind() in nn:
#             nn.remove(parent.ind())
#         parent = parent.parent
#
#     nn_no_collision = []
#     for [xi, yi, thetai] in nn:
#         if not conf_space.space[xi, yi, thetai]:
#             nn_no_collision.append((xi, yi, thetai))
#
#     return nn_no_collision