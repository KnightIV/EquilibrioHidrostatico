function plot_data_helper(x, y, x_label, y_label, fig_title, format, y_log_scale)
  if (y_log_scale)
    semilogy(x, y, format, "MarkerSize", 2, "LineWidth", 1)
  else
    plot(x, y, format, "MarkerSize", 2, "LineWidth", 1)
  endif
  grid on
  xlabel(x_label)
  ylabel(y_label)
  title(fig_title)
end
