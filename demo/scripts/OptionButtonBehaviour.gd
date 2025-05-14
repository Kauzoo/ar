extends OptionButton 

signal update_value(value : int) # Value is id of selected item
@onready var cvcamera = get_tree().get_first_node_in_group("cvcamera")

func _ready() -> void:
	if (self.selected == -1):
		printerr("@" + self.name + " has no default value assigned")
	if (update_value.has_connections() == false):
		printerr("@" + self.name + " is not connected")
	self.item_selected.connect(_on_item_selected)
	cvcamera.ready.connect(_on_camera_ready)
	pass

func _on_item_selected(index : int) -> void:
	update_value.emit(self.get_item_id(index))
	pass

func _on_camera_ready() -> void:
	update_value.emit(self.get_item_id(self.selected))
	pass
