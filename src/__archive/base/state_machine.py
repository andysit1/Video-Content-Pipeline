

class State():
    """
    Represents a game state interface.

    Attributes:
        engine: The game engine associated with the state.
    """

    def __init__(self, engine):
        """
        Initialize a State object.

        Args:
            engine: The game engine associated with the state.
        """
        self.engine = engine


    #draw and event is not really needed unless in the future we add new functions
    #for now can ignore.
    def on_draw(self, surface):
        pass

    def on_event(self, event):
        """
        Method called when an event occurs.

        Args:
            event: The event object.
        """
        pass

    def on_update(self):
        """
        Method called to update the state.

        Args:
            delta: The time elapsed since the last update.
        """
        pass


class Machine:
    """
    Manages transitions between different game states.
    """
    def __init__(self):
        """
        Initialize a Machine object.
        """
        self.current = None
        self.next_state = None

    def update(self):
        """
        Update the current state.
        """
        if self.next_state:
            self.current = self.next_state
            self.next_state = None


class Engine:
    def __init__(self):
        self.running = True
        self.machine = Machine()

    def loop(self):
        """
        Main game loop which handles all draw, update, on_event, and movement
        """
        while self.running:
            self.machine.update()
            self.machine.current.on_update()

            if self.machine.current.tag == "end":
                self.running = False


    def run(self, state):
        self.machine.current = state
        self.loop()

