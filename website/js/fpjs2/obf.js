function q(e, r) {
	return n(this, void 0, void 0, (function() {
		var n, o, a;
		return t(this, (function(t) {
			switch(t.label) {
				case 0:
					return function(e, n) {
						e.width = 240, e.height = 60, n.textBaseline = "alphabetic", n.fillStyle = "#f60", n.fillRect(100, 1, 62, 20), n.fillStyle = "#069", n.font = '11pt "Times New Roman"';
						var t = "Cwm fjordbank gly ".concat(String.fromCharCode(55357, 56835));
						n.fillText(t, 2, 15), n.fillStyle = "rgba(102, 204, 0, 0.2)", n.font = "18pt Arial", n.fillText(t, 4, 45)
					}(e, r), [4, i()];
				case 1:
					return t.sent(), n = $(e), o = $(e), n !== o ? [2, ["unstable", "unstable"]] : (function(e, n) {
						e.width = 122, e.height = 110, n.globalCompositeOperation = "multiply";
						for(var t = 0, r = [
								["#f2f", 40, 40],
								["#2ff", 80, 40],
								["#ff2", 60, 80]
							]; t < r.length; t++) {
							var o = r[t],
								i = o[0],
								a = o[1],
								c = o[2];
							n.fillStyle = i, n.beginPath(), n.arc(a, c, 40, 0, 2 * Math.PI, !0), n.closePath(), n.fill()
						}
						n.fillStyle = "#f9c", n.arc(60, 60, 60, 0, 2 * Math.PI, !0), n.arc(60, 60, 20, 0, 2 * Math.PI, !0), n.fill("evenodd")
					}(e, r), [4, i()]);
				case 2:
					return t.sent(), a = $(e), [2, [n, a]]
			}
		}))
	}))
}

const canvas = document.createElement('canvas')
canvas.width = 1
canvas.height = 1
context = canvas.getContext('2d')
q(canvas, context)