import pyglet
import os
import random

from PIL import Image
import glob

pyglet.resource.path = [(os.path.dirname(__file__) + '/resources')]
pyglet.resource.reindex()


class Render(pyglet.window.Window):
    def __init__(self, shared):
        self.shared_Elevator = shared
        self.elevator_num = shared.elenum
        self.num_floor = shared.number_of_floors
        self.screen_x = int(self.elevator_num * 50 + 300)
        self.screen_y = int(self.num_floor * 12.5 * 4 + 90)
        super(Render, self).__init__(width=self.screen_x, height=self.screen_y)
        self.create_window()
        self.frame = []
        self.image_count = 0
        self.gif_count = 0
        if not os.path.exists('./image_buffer'):
            os.makedirs('./image_buffer')

        if not os.path.exists('./animation_buffer'):
            os.makedirs('./animation_buffer')

    def create_window(self):
        self.set_size(self.screen_x, self.screen_y)
        self.set_visible()
        self.load_images()
        self.init_batch()

    def center_image(self, image):
        image.anchor_x = image.width // 2
        image.anchor_y = image.height // 2

    def load_images(self):
        self.man_image_1 = pyglet.resource.image('1.png')
        self.man_image_2 = pyglet.resource.image('2.png')
        self.man_image_3 = pyglet.resource.image('3.png')
        self.man_image_4 = pyglet.resource.image('4.png')
        self.human_image = [self.man_image_1, self.man_image_2, self.man_image_3, self.man_image_4]
        self.background = pyglet.resource.image("background.png")
        self.line = pyglet.resource.image("line.png")
        self.up = pyglet.resource.image("up.png")
        self.down = pyglet.resource.image("down.png")
        self.steady = pyglet.resource.image("steady.png")
        self.up_open = pyglet.resource.image("up_open.png")
        self.down_open = pyglet.resource.image("down_open.png")
        self.steady_open = pyglet.resource.image("steady_open.png")
        self.line.width, self.line.height = self.screen_x, 3
        self.center_image(self.line)
        self.background.width, self.background.height = self.screen_x, self.screen_y
        self.center_image(self.background)
        self.background = pyglet.sprite.Sprite(img=self.background, x=self.screen_x // 2, y=self.screen_y // 2)

        for image in self.human_image:
            image.width, image.height = 20, 38
            self.center_image(image)

        self.up.width, self.up.height = 35, 45
        self.center_image(self.up)

        self.down.width, self.down.height = 35, 45
        self.center_image(self.down)

        self.steady.width, self.steady.height = 35, 50
        self.center_image(self.steady)

        self.down_open.width, self.down_open.height = 35, 45
        self.center_image(self.down_open)

        self.up_open.width, self.up_open.height = 35, 45
        self.center_image(self.up_open)

        self.steady_open.width, self.steady_open.height = 35, 50
        self.center_image(self.steady_open)

        self.up_label = pyglet.text.Label(text="Waiting up", font_size=12, x=100, y=self.screen_y - 35,
                                          anchor_x='center')
        self.down_label = pyglet.text.Label(text="Waiting down", font_size=12, x=self.screen_x - 100,
                                            y=self.screen_y - 35, anchor_x='center')
        self.level_label = pyglet.text.Label(text="Elevator Simulator", font_size=12, x=self.screen_x // 2,
                                             y=self.screen_y - 15, anchor_x='center')

    def init_batch(self):
        """
        line_batch: lines to separate floors, which can be initialized here as it remains unchanged
        waiting_people_batch: people on the two sides
        elevator_batch: including elevator (square) and passengers (circle)
        """
        self.line_batch = pyglet.graphics.Batch()
        self.waiting_people_batch = pyglet.graphics.Batch()
        self.elevator_batch = list()
        self.line_ele = list()
        self.waiting_people_ele = list()
        self.elevator_ele = list()
        self.right_list = [[] for i in range(self.num_floor)]
        self.left_list = [[] for i in range(self.num_floor)]
        for i in range(self.elevator_num):
            self.elevator_batch.append(pyglet.graphics.Batch())

        for i in range(self.num_floor):
            self.line_ele.append(
                pyglet.sprite.Sprite(img=self.line, x=self.screen_x // 2, y=12.5 * 4 * i + 50,
                                     batch=self.line_batch))

    def update(self):
        # update waiting_people_batch
        waiting_up, waiting_down = self.shared_Elevator._wait_upward_persons_queue , self.shared_Elevator._wait_downward_persons_queue
        for ele in self.waiting_people_ele:
            ele.delete()
        self.waiting_people_ele = []
        for i in range(self.num_floor):
            # left side, waiting up people
            if len(waiting_up[i]) > 9:
                self.waiting_people_ele.append(
                    pyglet.sprite.Sprite(img=self.man_image_1, x=100, y=12.5 * 4 * i + 20,
                                         batch=self.waiting_people_batch))
                self.waiting_people_ele.append(pyglet.text.Label(text="x {}".format(len(waiting_up[i])), font_size=8,
                                                                 x=130, y=12.5 * 4 * i + 15,
                                                                 anchor_x='center', batch=self.waiting_people_batch))
            else:
                while len(self.left_list[i]) < len(waiting_up[i]):
                    self.left_list[i].append(random.randint(0, 3))
                while len(self.left_list[i]) > len(waiting_up[i]):
                    self.left_list[i].pop(0)
                for j in range(len(waiting_up[i])):
                    self.waiting_people_ele.append(
                        pyglet.sprite.Sprite(img=self.human_image[self.left_list[i][j]], x=140 - 15 * j,
                                             y=12.5 * 4 * i + 22, batch=self.waiting_people_batch))
            # right side, waiting down people
            if len(waiting_down[i]) > 9:
                self.waiting_people_ele.append(pyglet.sprite.Sprite(img=self.man_image_1, x=self.screen_x - 125,
                                                                    y=12.5 * 4 * i + 20,
                                                                    batch=self.waiting_people_batch))
                self.waiting_people_ele.append(pyglet.text.Label(text="x {}".format(len(waiting_down[i])), font_size=8,
                                                                 x=self.screen_x - 95,
                                                                 y=12.5 * 4 * i + 15, anchor_x='center',
                                                                 batch=self.waiting_people_batch))
            else:
                while len(self.right_list[i]) < len(waiting_down[i]):
                    self.right_list[i].append(random.randint(0, 3))
                while len(self.right_list[i]) > len(waiting_down[i]):
                    self.right_list[i].pop(0)
                for j in range(len(waiting_down[i])):
                    self.waiting_people_ele.append(pyglet.sprite.Sprite(img=self.human_image[self.right_list[i][j]],
                                                                        x=self.screen_x - 140 + 15 * j,
                                                                        y=12.5 * 4 * i + 22,
                                                                        batch=self.waiting_people_batch))

        # update elevator_batch
        for ele in self.elevator_ele:
            ele.delete()
        self.elevator_ele = []
        for i in range(self.elevator_num):
            self.elevator_floor = self.shared_Elevator.elevator.ele_cur_position
            if self.shared_Elevator.elevator.cur_move_direction[i]  == 1:
                if self.shared_Elevator.elevator.is_opening[i]:
                    self.elevator_ele.append(pyglet.sprite.Sprite(img=self.up_open, x=175 + i * 50,
                                                              y=self.elevator_floor[i] * 4 * 12.5 - 25,
                                                              batch=self.elevator_batch[i]))
                elif not self.shared_Elevator.elevator.is_opening[i]:
                    self.elevator_ele.append(pyglet.sprite.Sprite(img=self.up, x=175 + i * 50,
                                                              y=self.elevator_floor[i] * 4 * 12.5 - 25,
                                                              batch=self.elevator_batch[i]))
            elif self.shared_Elevator.elevator.cur_move_direction[i]  == -1:
                if self.shared_Elevator.elevator.is_opening[i]:
                    self.elevator_ele.append(pyglet.sprite.Sprite(img=self.down_open, x=175 + i * 50,
                                                              y=self.elevator_floor[i] * 4 * 12.5 - 25,
                                                              batch=self.elevator_batch[i]))
                elif not self.shared_Elevator.elevator.is_opening[i]:
                    self.elevator_ele.append(pyglet.sprite.Sprite(img=self.down, x=175 + i * 50,
                                                                y=self.elevator_floor[i] * 4 * 12.5 - 25,
                                                                batch=self.elevator_batch[i]))
            else:
                if self.shared_Elevator.elevator.is_opening[i]:
                    self.elevator_ele.append(pyglet.sprite.Sprite(img=self.steady_open, x=175 + i * 50,
                                                              y=self.elevator_floor[i] * 4 * 12.5 - 25,
                                                              batch=self.elevator_batch[i]))
                # elif not self.shared_Elevator.elevator.door_is_open[i]:
                else:
                    self.elevator_ele.append(pyglet.sprite.Sprite(img=self.steady, x=175 + i * 50,
                                                                y=self.elevator_floor[i] * 4 * 12.5 - 25,
                                                                batch=self.elevator_batch[i]))

                # when too many passengers in an elevator, use numeric numbers to show
            self.elevator_ele.append(pyglet.text.Label(text="loading", font_size=8,
                                                       x=175 + i * 50, y=self.screen_y - 45, anchor_x='center',
                                                       batch=self.elevator_batch[i]))
            self.elevator_ele.append(
                pyglet.text.Label(text="{} people".format(self.shared_Elevator.elevator.cur_loading[i]), font_size=8,
                                  x=175 + i * 50, y=self.screen_y - 58, anchor_x='center',
                                  batch=self.elevator_batch[i]))
        # pyglet.gl.set_current()

    def view(self):
        self.clear()
        self.update()

        self.dispatch_events()
        self.background.draw()
        self.line_batch.draw()
        self.level_label.draw()
        self.up_label.draw()
        self.down_label.draw()
        self.waiting_people_batch.draw()
        for i in range(self.elevator_num):
            self.elevator_batch[i].draw()

        # # save the current window image
        pyglet.image.get_buffer_manager().get_color_buffer().save(
            './image_buffer/image{}.png'.format(str(self.image_count)))
        image = glob.glob("./image_buffer/image{}.png".format(str(self.image_count)))
        new_frame = Image.open(image[0])
        self.frame.append(new_frame)
        self.image_count += 1

        # print the window
        self.flip()

        if self.image_count == 500:
        # combine images in self.frame to gif and save
            print("Creating gif animation, please wait")
            self.frame[0].save('./animation_buffer/animation{}.gif'.format(str(self.gif_count)),
                                format='GIF', append_images=self.frame[1:],
                                save_all=True, duration=100, loop=0)
            self.frame = []
            self.image_count = 0
        self.gif_count += 1
