import React from "react";
import {
  List,
  Info,
  DotsThreeVertical,
  ChatCircle,
  Plus,
  Cube,
  Flask,
  Books,
  ChartBar,
  Scroll,
  type IconProps as PhosphorIconProps,
} from "phosphor-react";

type Props = PhosphorIconProps & { className?: string };

// Helper to set duotone + allow custom classes/colors per use
const wrap =
  (Comp: React.ComponentType<PhosphorIconProps>) =>
  ({ className, size = 18, color = "currentColor", ...rest }: Props) =>
    <Comp weight="duotone" size={size} color={color} className={className} {...rest} />;

export const Icons = {
  // Header
  Menu: wrap(List),
  Info: wrap(Info),
  More: wrap(DotsThreeVertical),

  // Left rail
  New: wrap(Plus),
  Chat: wrap(ChatCircle),
  Models: wrap(Cube),
  Train: wrap(Flask),
  Library: wrap(Books),
  Results: wrap(ChartBar),
  Logs: wrap(Scroll),
};