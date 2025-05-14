extends Sprite2D

@onready var debug_canvas : Sprite2D = self.get_node(^"DebugSprite")

func _ready() -> void:
	debug_canvas.visible = false
