extends Control

var slider : VSlider;
var textbox : LineEdit;


func _ready() -> void:
	textbox = self.get_node(^"LineEdit");
	slider = self.get_node(^"VSlider");
	textbox.text_submitted.connect(_text_changed);
	slider.value_changed.connect(_slider_changed)
	pass;


func _text_changed(new_text : String) -> void:
	var val = float(new_text)
	if (val >= slider.min_value && val <= slider.max_value):
		slider.value = val;
	pass

func _slider_changed(value : float) -> void:
	textbox.text = str(value);
	pass
