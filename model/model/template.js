
$(document).ready(draw_arcs)

function join(elm1, elm2, sentence, is_downward) {
	if(elm1.offset().left > elm2.offset().left) {
		var temp = elm1
		elm1 = elm2
		elm2 = temp
	}
	var top = elm1.offset().top
	var left = elm1.offset().left
	var arc_width = get_arc_width(elm1, elm2)
	var new_arc = $('<img src="arc.svg" />')

	if(is_downward) {
		new_arc.css({
			"position": "absolute",
			"top": "16px",
			"left": elm1.width() / 2,
			"width": arc_width,
			"transform": "scaleY(-1)"
		})
	} else {
		new_arc.css({
			"position": "absolute",
			"top": -arc_width/2,
			"left": elm1.width() / 2,
			"width": arc_width
		})
	}
	elm1.append(new_arc)
	arc_height = arc_width / 2
	return arc_height
}


function get_arc_width(elm1, elm2) {
	// Calculates arc width such that arc's endpoints land on token midpoints.
	var midpoint1 = elm1.offset().left + elm1.width() / 2
	var midpoint2 = elm2.offset().left + elm2.width() / 2
	return midpoint2 - midpoint1
}


function draw_arcs() {
	var sentences = $('p')
	sentences.each(function(){
		sentence = $(this)
		var words = sentence.find('word')

		// Join the words.  Keep track of the maximum height of arc drawn.
		var max_arc_height = 0
		var max_arc_depth = 0
		words.each(function(word_idx){
			var elm = $(this)
			var head = words.eq(elm.attr("head"))
			var arc_height = join(elm, head, sentence, false)
			max_arc_height = Math.max(arc_height, max_arc_height)

			var alt_head_ptr = elm.attr("alt-head")
			if(alt_head_ptr !== undefined) {
				var alt_head = words.eq(alt_head_ptr)
				var arc_depth = join(elm, alt_head, sentence, true)
				max_arc_depth = Math.max(arc_depth, max_arc_depth)
			}
		})

		// Adjust this sentence's padding based on the max arc height.
		$(this).css({
			'padding-top': max_arc_height,
			'padding-bottom': max_arc_depth
		})
	})
}

