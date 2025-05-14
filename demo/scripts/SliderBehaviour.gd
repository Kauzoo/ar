extends Control

@onready var slider : VSlider = self.get_node(^"VSlider")
@onready var textbox : LineEdit = get_node(^"LineEdit")
signal update_value(value : float)
@onready var cvcamera = get_tree().get_first_node_in_group("cvcamera")

func _ready() -> void:
	if (update_value.has_connections() == false):
		printerr("@" + self.name + " is not connected")
	textbox.text_submitted.connect(_text_changed)
	slider.value_changed.connect(_slider_changed)
	cvcamera.ready.connect(_on_camera_ready)
	textbox.text = str(slider.value)
	pass


func _text_changed(new_text : String) -> void:
	var val = float(new_text)
	if (val >= slider.min_value && val <= slider.max_value):
		slider.value = val;
	pass

func _slider_changed(value : float) -> void:
	textbox.text = str(value);
	update_value.emit(value)
	pass

func _on_camera_ready() -> void:
	update_value.emit(slider.value)
	pass
