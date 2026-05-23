"""
Debug7: Trace the full attacks pipeline for id_src=20 at step 64.
Compare pandas vs polars step by step.
"""
import kaggle_environments as ke
import math, copy, random
import numpy as np
import pandas as pd
import polars as pl

CENTER = 50.0; SUN_RADIUS = 10.0; ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0; NB_STEPS_SIM = 10; PLANET_MARGIN = 0.1
BOARD_SIZE = 100.0; MAX_NB_STEP = 500

class Obs:
    def __init__(self, planets, initial_planets=None, fleets=None,
                 next_fleet_id=100, comets=None, comet_planet_ids=None, angular_velocity=0.0):
        self.planets = [list(p) for p in planets]
        self.initial_planets = [list(p) for p in (initial_planets if initial_planets is not None else planets)]
        self.fleets = [list(f) for f in (fleets or [])]
        self.next_fleet_id = next_fleet_id; self.comets = comets or []
        self.comet_planet_ids = comet_planet_ids or []; self.angular_velocity = angular_velocity

def distance(p1, p2): return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
def point_to_segment_distance(p, v, w):
    l2=(v[0]-w[0])**2+(v[1]-w[1])**2
    if l2==0: return distance(p,v)
    t=max(0,min(1,((p[0]-v[0])*(w[0]-v[0])+(p[1]-v[1])*(w[1]-v[1]))/l2))
    return distance(p,(v[0]+t*(w[0]-v[0]),v[1]+t*(w[1]-v[1])))

def interpreter(obs, actions, step, num_agents=2):
    obs0 = obs
    expired = set()
    for group in obs0.comets:
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            if idx >= len(group["paths"][i]): expired.add(pid)
    if expired:
        obs0.planets = [p for p in obs0.planets if p[0] not in expired]
        obs0.initial_planets = [p for p in obs0.initial_planets if p[0] not in expired]
        obs0.comet_planet_ids = [pid for pid in obs0.comet_planet_ids if pid not in expired]
        for g in obs0.comets: g["planet_ids"] = [pid for pid in g["planet_ids"] if pid not in expired]
        obs0.comets = [g for g in obs0.comets if g["planet_ids"]]
    def process_moves(pid, action):
        if not action or not isinstance(action, list): return
        for move in action:
            if len(move) != 3: continue
            from_id, angle, ships = move; ships = int(ships)
            fp = next((p for p in obs0.planets if p[0] == from_id), None)
            if fp and fp[1] == pid and fp[5] >= ships and ships > 0:
                fp[5] -= ships
                obs0.fleets.append([obs0.next_fleet_id, pid, fp[2]+math.cos(angle)*(fp[4]+0.1), fp[3]+math.sin(angle)*(fp[4]+0.1), angle, from_id, ships])
                obs0.next_fleet_id += 1
    for i in range(num_agents): process_moves(i, actions[i])
    for p in obs0.planets:
        if p[1] != -1: p[5] += p[6]
    fr = []; combat = {p[0]: [] for p in obs0.planets}
    for f in obs0.fleets:
        sp = min(1.0+(MAX_SPEED-1.0)*(math.log(f[6])/math.log(1000))**1.5, MAX_SPEED)
        op = (f[2], f[3]); f[2] += math.cos(f[4])*sp; f[3] += math.sin(f[4])*sp; np2 = (f[2], f[3]); hit = False
        for p in obs0.planets:
            if point_to_segment_distance((p[2],p[3]),op,np2) < p[4]: combat[p[0]].append(f);fr.append(f);hit=True;break
        if hit: continue
        if not(0<=f[2]<=BOARD_SIZE and 0<=f[3]<=BOARD_SIZE): fr.append(f); continue
        if point_to_segment_distance((CENTER,CENTER),op,np2) < SUN_RADIUS: fr.append(f); continue
    cps = set(obs0.comet_planet_ids); ibi = {p[0]: p for p in obs0.initial_planets}
    def sw(planet, op, np2):
        if op == np2: return
        for f in obs0.fleets:
            if f not in fr and point_to_segment_distance((f[2],f[3]),op,np2) < planet[4]: combat[planet[0]].append(f);fr.append(f)
    for p in obs0.planets:
        if p[0] in cps: continue
        ip = ibi.get(p[0])
        if not ip: continue
        dx=ip[2]-CENTER;dy=ip[3]-CENTER;r=math.sqrt(dx**2+dy**2);op=(p[2],p[3])
        if r+p[4] < ROTATION_RADIUS_LIMIT:
            ia=math.atan2(dy,dx);p[2]=CENTER+r*math.cos(ia+obs0.angular_velocity*step);p[3]=CENTER+r*math.sin(ia+obs0.angular_velocity*step)
        sw(p,op,(p[2],p[3]))
    expired2 = set()
    for group in obs0.comets:
        group["path_index"] += 1; idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            p = next((x for x in obs0.planets if x[0]==pid),None)
            if not p: continue
            pp = group["paths"][i]
            if idx >= len(pp): expired2.add(pid)
            else:
                op=(p[2],p[3]);p[2]=pp[idx][0];p[3]=pp[idx][1]
                if op[0]>=0: sw(p,op,(p[2],p[3]))
    if expired2:
        obs0.planets=[p for p in obs0.planets if p[0] not in expired2]
        obs0.initial_planets=[p for p in obs0.initial_planets if p[0] not in expired2]
        obs0.comet_planet_ids=[pid for pid in obs0.comet_planet_ids if pid not in expired2]
        for g in obs0.comets: g["planet_ids"]=[pid for pid in g["planet_ids"] if pid not in expired2]
        obs0.comets=[g for g in obs0.comets if g["planet_ids"]]
    obs0.fleets=[f for f in obs0.fleets if f not in fr]
    for pid,pf in combat.items():
        p=next((x for x in obs0.planets if x[0]==pid),None)
        if not p or not pf: continue
        ps={}
        for f in pf: ps[f[1]]=ps.get(f[1],0)+f[6]
        if not ps: continue
        sp2=sorted(ps.items(),key=lambda x:x[1],reverse=True);tp,ts=sp2[0]
        if len(sp2)>1: ss=ts-sp2[1][1];so=tp if ss>0 else -1
        else: so=tp;ss=ts
        if ss>0:
            if p[1]==so: p[5]+=ss
            else: p[5]-=ss;p[1]=so if p[5]<0 else p[1];p[5]=abs(p[5]) if p[5]<0 else p[5]
    obs0.fleets=[f for f in obs0.fleets if f not in fr]
    return {"planets":obs0.planets,"initial_planets":obs0.initial_planets,"fleets":obs0.fleets,"next_fleet_id":obs0.next_fleet_id,"comets":obs0.comets,"comet_planet_ids":obs0.comet_planet_ids}

def _simulate(obs, global_step, num_agents, n_steps=NB_STEPS_SIM):
    sim=copy.deepcopy(obs);na=[[]for _ in range(num_agents)];rows=[]
    for i in range(n_steps+1):
        for p in sim.planets:
            r=math.hypot(p[2]-CENTER,p[3]-CENTER)
            nat="comet" if p[0] in sim.comet_planet_ids else ("moving" if r+p[4]<ROTATION_RADIUS_LIMIT else "fix")
            rows.append({"step":global_step+i,"id":p[0],"x":p[2],"y":p[3],"radius":p[4],"ships":p[5],"production":p[6],"owner":p[1],"nature":nat})
        interpreter(sim,na,global_step+i,num_agents)
    return pd.DataFrame(rows)

class IntervalProcessor:
    @staticmethod
    def merge_intervals(intervals):
        if not intervals: return []
        intervals=sorted(intervals,key=lambda x:x[0]);merged=[intervals[0]]
        for current in intervals[1:]:
            pm,px=merged[-1];cm,cx=current
            if cm<=px: merged[-1]=(pm,max(px,cx))
            else: merged.append(current)
        return merged
    @staticmethod
    def create_cumulative_obstacles(possible_attacks,min_step=0):
        max_step=int(possible_attacks["step"].max());attack_map={}
        for (id_src,ship,step),group in possible_attacks.groupby(["id_src","ships_sent","step"]):
            unwrapped=[]
            for _,row in group.iterrows():
                amin,amax=row["angle_min"],row["angle_max"]
                if amin>amax: unwrapped.extend([(amin,2*np.pi),(0.0,amax)])
                else: unwrapped.append((amin,amax))
            attack_map[(id_src,ship,step)]=unwrapped
        unique_combos=possible_attacks[["id_src","ships_sent"]].drop_duplicates().values;records=[]
        for id_src,ship in unique_combos:
            current_intervals=[];merged=[]
            for step in range(min_step,max_step+1):
                if (id_src,ship,step) in attack_map:
                    current_intervals.extend(attack_map[(id_src,ship,step)])
                    merged=IntervalProcessor.merge_intervals(current_intervals)
                records.append({"step":step+1,"id_src":id_src,"ships_sent":ship,"obstacle_list":merged})
        return pd.DataFrame(records)
    @staticmethod
    def subtract_intervals(target_min,target_max,blocked_intervals):
        safe=[(target_min,target_max)]
        for b_min,b_max in blocked_intervals:
            next_safe=[]
            for s_min,s_max in safe:
                if b_max<=s_min or b_min>=s_max: next_safe.append((s_min,s_max))
                else:
                    if b_min>s_min: next_safe.append((s_min,b_min))
                    if b_max<s_max: next_safe.append((b_max,s_max))
            safe=next_safe
            if not safe: break
        return safe
    @staticmethod
    def compute_free_angles(row):
        amin,amax,obstacles=row["angle_min"],row["angle_max"],row["obstacle_list"]
        if not isinstance(obstacles,list) or not obstacles: return[(amin,amax)]
        targets=[(amin,2*np.pi),(0.0,amax)] if amin>amax else [(amin,amax)]
        all_free=[]
        for t_min,t_max in targets: all_free.extend(IntervalProcessor.subtract_intervals(t_min,t_max,obstacles))
        has_end=any(abs(f[1]-2*np.pi)<1e-9 for f in all_free);has_start=any(abs(f[0])<1e-9 for f in all_free)
        if has_end and has_start and len(all_free)>1:
            ei=next(i for i,f in enumerate(all_free) if abs(f[1]-2*np.pi)<1e-9)
            si=next(i for i,f in enumerate(all_free) if abs(f[0])<1e-9)
            w=(all_free[ei][0],all_free[si][1])
            all_free=[f for i,f in enumerate(all_free) if i not in {ei,si}];all_free.append(w)
        return all_free
    @staticmethod
    def interval_to_final_angle(series):
        def process(intervals):
            if not isinstance(intervals,list) or not intervals: return np.nan
            widest=-1.0;best=np.nan
            for amin,amax in intervals:
                span=amax-amin if amin<=amax else (2*np.pi-amin)+amax
                if span>widest:
                    widest=span;mid=(amin+amax)/2 if amin<=amax else amin+span/2;best=mid%(2*np.pi)
            return best
        return series.map(process)


class IntervalProcessorPolars:
    @staticmethod
    def merge_intervals(intervals):
        if not intervals: return []
        intervals=sorted(intervals,key=lambda x:x[0]);merged=[list(intervals[0])]
        for current in intervals[1:]:
            pm,px=merged[-1];cm,cx=current
            if cm<=px: merged[-1]=[pm,max(px,cx)]
            else: merged.append(list(current))
        return [tuple(x) for x in merged]
    @staticmethod
    def subtract_intervals(target_min,target_max,blocked_intervals):
        safe=[(target_min,target_max)]
        for b_min,b_max in blocked_intervals:
            next_safe=[]
            for s_min,s_max in safe:
                if b_max<=s_min or b_min>=s_max: next_safe.append((s_min,s_max))
                else:
                    if b_min>s_min: next_safe.append((s_min,b_min))
                    if b_max<s_max: next_safe.append((b_max,s_max))
            safe=next_safe
            if not safe: break
        return safe
    @staticmethod
    def create_cumulative_obstacles(possible_attacks,min_step=0):
        max_step=int(possible_attacks["step"].max());attack_map={}
        for row in possible_attacks.select(["id_src","ships_sent","step","angle_min","angle_max"]).to_dicts():
            key=(row["id_src"],row["ships_sent"],row["step"]);amin,amax=row["angle_min"],row["angle_max"]
            if amin>amax: attack_map.setdefault(key,[]).extend([(amin,2*np.pi),(0.0,amax)])
            else: attack_map.setdefault(key,[]).append((amin,amax))
        unique_combinations=possible_attacks.select(["id_src","ships_sent"]).unique(maintain_order=True).to_numpy()
        steps_col,id_srcs_col,ships_col,obstacles_col=[],[],[],[]
        for id_src,ship in unique_combinations:
            current_intervals=[];merged=[]
            for step in range(min_step,max_step+1):
                if (id_src,ship,step) in attack_map:
                    current_intervals.extend(attack_map[(id_src,ship,step)])
                    merged=IntervalProcessorPolars.merge_intervals(current_intervals)
                steps_col.append(step+1);id_srcs_col.append(id_src);ships_col.append(ship)
                obstacles_col.append([[a,b] for a,b in merged])
        return pl.DataFrame({"step":steps_col,"id_src":id_srcs_col,"ships_sent":ships_col,"obstacle_list":obstacles_col},
            schema={"step":pl.Int64,"id_src":pl.Int64,"ships_sent":pl.Int64,"obstacle_list":pl.List(pl.List(pl.Float64))})
    @staticmethod
    def compute_free_angles(row):
        if hasattr(row,"as_py"): row=row.as_py()
        amin,amax=row["angle_min"],row["angle_max"];raw_obs=row["obstacle_list"]
        if hasattr(raw_obs,"to_list"): raw_obs=raw_obs.to_list()
        obstacles=raw_obs or []
        if not obstacles: return [[amin,amax]]
        targets=[(amin,2*np.pi),(0.0,amax)] if amin>amax else [(amin,amax)]
        all_free=[]
        for t_min,t_max in targets: all_free.extend(IntervalProcessorPolars.subtract_intervals(t_min,t_max,obstacles))
        has_end=any(abs(f[1]-2*np.pi)<1e-9 for f in all_free);has_start=any(abs(f[0])<1e-9 for f in all_free)
        if has_end and has_start and len(all_free)>1:
            ei=next(i for i,f in enumerate(all_free) if abs(f[1]-2*np.pi)<1e-9)
            si=next(i for i,f in enumerate(all_free) if abs(f[0])<1e-9)
            wrapped=(all_free[ei][0],all_free[si][1])
            all_free=[f for i,f in enumerate(all_free) if i not in {ei,si}];all_free.append(wrapped)
        return [[a,b] for a,b in all_free]
    @staticmethod
    def interval_to_final_angle(series):
        def _best(intervals):
            if intervals is None: return float("nan")
            if hasattr(intervals,"to_list"): intervals=intervals.to_list()
            if not intervals: return float("nan")
            widest,best=-1.0,float("nan")
            for interval in intervals:
                amin,amax=interval[0],interval[1];span=(amax-amin) if amin<=amax else (2*np.pi-amin+amax)
                if span>widest:
                    widest=span;mid=(amin+amax)/2 if amin<=amax else amin+span/2;best=mid%(2*np.pi)
            return best
        return series.map_elements(_best,return_dtype=pl.Float64)


def random_agent_fn(obs):
    player=obs.player;my=[p for p in obs.planets if p[1]==player]
    if not my: return []
    p=random.choice(my);ships=p[5]//2
    if ships<1: return []
    return [[p[0],random.uniform(0,2*math.pi),ships]]


SEED=42;N_STEPS=100
random.seed(SEED)
env=ke.make("orbit_wars",debug=False)
env.reset(2)

for env_step in range(N_STEPS):
    obs0=env.state[0].observation;obs1=env.state[1].observation
    df=_simulate(copy.deepcopy(obs0),global_step=env_step,num_agents=2,n_steps=NB_STEPS_SIM)

    # We need take_action result to step correctly
    # Replicate take_action pandas inline to get intermediate steps
    player_id = 0

    if env_step == 64:
        print(f"=== Step 64: full pipeline trace for id_src=20 ===")

        # ── PANDAS pipeline ──────────────────────────────────────────────────
        mine_pd = (
            df.assign(is_mine=lambda d:(d["owner"]==player_id).astype(int)).groupby("id")
            .agg(step_src=("step","first"),x_src=("x","first"),y_src=("y","first"),radius_src=("radius","first"),
                 ships_min=("ships","min"),production_src=("production","first"),nature_src=("nature","first"),
                 owner_src=("owner","first"),row_count=("ships","size"),is_mine=("is_mine","sum"))
            .query("row_count == is_mine and owner_src==@player_id").reset_index(drop=False).rename(columns={"id":"id_src"})
        )
        expanded_pd = (
            mine_pd.assign(ships_sent=(mine_pd["ships_min"]+mine_pd["production_src"]*NB_STEPS_SIM).apply(lambda n:list(range(1,n+1))))
            .explode("ships_sent").astype({"ships_sent":int}).reset_index(drop=True)
        )
        df_st_pd = expanded_pd.merge(df,how="cross").query("step > step_src and id != id_src")
        pa_pd = (
            df_st_pd.assign(
                dist_tgt_src=lambda d:((d["x"]-d["x_src"])**2+(d["y"]-d["y_src"])**2)**0.5,
                step_diff=lambda d:d["step"]-d["step_src"],
                fleet_speed=lambda d:1.0+(MAX_SPEED-1.0)*(np.log(d["ships_sent"])/math.log(1000))**1.5,
                dist_fleet_src_min=lambda d:d["step_diff"]*d["fleet_speed"]+PLANET_MARGIN+d["radius_src"],
                dist_fleet_src_max=lambda d:(d["step_diff"]+1)*d["fleet_speed"]+PLANET_MARGIN+d["radius_src"],
                collision=lambda d:((d["dist_tgt_src"]-d["radius"]<d["dist_fleet_src_min"])&(d["dist_fleet_src_min"]<d["dist_tgt_src"]+d["radius"]))|((d["dist_tgt_src"]-d["radius"]<d["dist_fleet_src_max"])&(d["dist_fleet_src_max"]<d["dist_tgt_src"]+d["radius"])))
            .query("collision")
            .assign(crossing_sun=lambda d:d.apply(lambda row:point_to_segment_distance((CENTER,CENTER),(row["x_src"],row["y_src"]),(row["x"],row["y"]))<SUN_RADIUS+PLANET_MARGIN,axis=1).astype(bool))
            .query("not crossing_sun")
            .assign(
                angle=lambda d:np.arctan2(d["y"]-d["y_src"],d["x"]-d["x_src"]),
                radius_angle=lambda d:np.maximum(
                    np.arccos(((d["dist_tgt_src"]**2+d["dist_fleet_src_min"]**2-d["radius"]**2)/(2*d["dist_tgt_src"]*d["dist_fleet_src_min"])).clip(-1,1)),
                    np.arccos(((d["dist_tgt_src"]**2+d["dist_fleet_src_max"]**2-d["radius"]**2)/(2*d["dist_tgt_src"]*d["dist_fleet_src_max"])).clip(-1,1))
                ),
                angle_min=lambda d:np.mod(d["angle"]-d["radius_angle"],2*math.pi),
                angle_max=lambda d:np.mod(d["angle"]+d["radius_angle"],2*math.pi),
            ).sort_values("step",ascending=True)
        )
        df_obs_pd = IntervalProcessor.create_cumulative_obstacles(pa_pd)
        awa_pd = (
            pa_pd.merge(df_obs_pd,how="left",on=["id_src","step","ships_sent"])
            .assign(angle_list=lambda d:d.apply(IntervalProcessor.compute_free_angles,axis=1))
            .query("angle_list.str.len() > 0")
        )

        # top-5 BEFORE comet filter (this is what pandas does)
        top5_pd_before_comet = (
            awa_pd.sort_values(["step","ships_sent"],ascending=True)
            .groupby(["id_src","id"],as_index=False).first()
            .sort_values(["step","ships_sent"],ascending=True)
            .groupby("id_src",as_index=False).head(5)[["id_src","id"]]
        )
        top5_20_before = top5_pd_before_comet[top5_pd_before_comet["id_src"]==20]["id"].tolist()
        print(f"\nPD top-5 for id_src=20 (before comet filter): ids={sorted(top5_20_before)}")

        # comet filter
        awa_comets_pd = awa_pd.query("nature_src == 'comet'")
        print(f"PD comet sources: {sorted(awa_comets_pd['id_src'].unique()) if not awa_comets_pd.empty else []}")
        if not awa_comets_pd.empty and max(max((awa_comets_pd["x_src"]-CENTER).abs()),max((awa_comets_pd["y_src"]-CENTER).abs()))>45:
            id_to_avoid=awa_comets_pd["id_src"].unique().tolist()
            print(f"PD avoiding comet sources: {id_to_avoid}")
            awa_pd_filtered=awa_pd.query("id_src not in @id_to_avoid")
        else:
            awa_pd_filtered=awa_pd
        print(f"PD awa after comet filter: {len(awa_pd_filtered)} rows")

        # Now the attacks step
        # key: planet_id_top_5_id_src was computed BEFORE comet filter, but merged with FILTERED awa
        attacks_pd = (
            top5_pd_before_comet
            .merge(awa_pd_filtered,how="left",on=["id_src","id"])
            .query("@player_id != owner")
            .assign(ships_needed=lambda d:np.where(d["owner"]==-1,d["ships"],d["ships"]+d["production"]))
            .query("ships_needed + 1 <= ships_sent and ships_sent <= ships_needed + production_src + 1")
            .sort_values(["step","ships_sent"],ascending=True).groupby(["id_src","id"],as_index=False).first()
            .assign(time_cost=lambda d:d["ships_needed"]/d["production_src"],
                    total_time_cost=lambda d:d.groupby("id_src")["time_cost"].transform("sum"),
                    score=lambda d:(d["total_time_cost"]-d["time_cost"]-d["step_diff"])*d["production"])
            .sort_values("score",ascending=False).groupby("id_src",as_index=False).first()
            .query("ships_sent <= ships_min")
        )
        print(f"\nPD attacks (before final_angle):")
        print(attacks_pd[["id_src","id","ships_sent","step","ships_min","score"]].to_string())

        # ── POLARS pipeline ──────────────────────────────────────────────────
        df_lf = pl.from_pandas(df).sort("step").lazy()
        mine_lz = (
            df_lf.with_columns(pl.when(pl.col("owner")==player_id).then(1).otherwise(0).alias("is_mine"))
            .group_by("id",maintain_order=True)
            .agg(pl.first("step").alias("step_src"),pl.first("x").alias("x_src"),pl.first("y").alias("y_src"),
                 pl.first("radius").alias("radius_src"),pl.min("ships").alias("ships_min"),
                 pl.first("production").alias("production_src"),pl.first("nature").alias("nature_src"),
                 pl.first("owner").alias("owner_src"),pl.len().alias("row_count"),pl.sum("is_mine").alias("is_mine"))
            .filter((pl.col("row_count")==pl.col("is_mine"))&(pl.col("owner_src")==player_id))
            .rename({"id":"id_src"}).collect()
        )
        dx_vw=pl.col("x")-pl.col("x_src");dy_vw=pl.col("y")-pl.col("y_src");l2=dx_vw.pow(2)+dy_vw.pow(2)
        dot=(CENTER-pl.col("x_src"))*dx_vw+(CENTER-pl.col("y_src"))*dy_vw
        t=(dot/pl.when(l2==0).then(pl.lit(1.0)).otherwise(l2)).clip(0.0,1.0)
        dsp=((CENTER-(pl.col("x_src")+t*dx_vw)).pow(2)+(CENTER-(pl.col("y_src")+t*dy_vw)).pow(2)).sqrt()
        dsd=((CENTER-pl.col("x_src")).pow(2)+(CENTER-pl.col("y_src")).pow(2)).sqrt()
        dts=pl.when(l2==0).then(dsd).otherwise(dsp)
        cs_expr=dts<(SUN_RADIUS+PLANET_MARGIN)
        dte=((pl.col("x")-pl.col("x_src")).pow(2)+(pl.col("y")-pl.col("y_src")).pow(2)).sqrt()
        sde=pl.col("step")-pl.col("step_src")
        fse=1.0+(MAX_SPEED-1.0)*(pl.col("ships_sent").cast(pl.Float64).log(base=math.e)/math.log(1000.0)).pow(1.5)
        dmine=sde*fse+PLANET_MARGIN+pl.col("radius_src");dmaxe=(sde+1)*fse+PLANET_MARGIN+pl.col("radius_src")
        cole=((dte-pl.col("radius")<dmine)&(dmine<dte+pl.col("radius")))|((dte-pl.col("radius")<dmaxe)&(dmaxe<dte+pl.col("radius")))
        pa_lz = (
            mine_lz.lazy()
            .with_columns(pl.int_ranges(1,pl.col("ships_min")+pl.col("production_src")*NB_STEPS_SIM+1,dtype=pl.Int64).alias("ships_sent"))
            .explode("ships_sent").join(df_lf,how="cross")
            .filter((pl.col("step")>pl.col("step_src"))&(pl.col("id")!=pl.col("id_src")))
            .with_columns([dte.alias("dist_tgt_src"),sde.alias("step_diff"),fse.alias("fleet_speed"),dmine.alias("dist_fleet_src_min"),dmaxe.alias("dist_fleet_src_max"),cole.alias("collision")])
            .filter(pl.col("collision")).with_columns(cs_expr.alias("crossing_sun")).filter(~pl.col("crossing_sun"))
            .with_columns(pl.arctan2(pl.col("y")-pl.col("y_src"),pl.col("x")-pl.col("x_src")).alias("angle"))
            .with_columns(pl.max_horizontal(
                ((pl.col("dist_tgt_src").pow(2)+pl.col("dist_fleet_src_min").pow(2)-pl.col("radius").pow(2))/(2*pl.col("dist_tgt_src")*pl.col("dist_fleet_src_min"))).clip(-1.0,1.0).arccos(),
                ((pl.col("dist_tgt_src").pow(2)+pl.col("dist_fleet_src_max").pow(2)-pl.col("radius").pow(2))/(2*pl.col("dist_tgt_src")*pl.col("dist_fleet_src_max"))).clip(-1.0,1.0).arccos(),
            ).alias("radius_angle"))
            .with_columns([((pl.col("angle")-pl.col("radius_angle"))%(2*math.pi)).alias("angle_min"),((pl.col("angle")+pl.col("radius_angle"))%(2*math.pi)).alias("angle_max")])
            .sort("step").collect()
        )
        df_obs_lz = IntervalProcessorPolars.create_cumulative_obstacles(pa_lz)
        awa_lz = (
            pa_lz.lazy().join(df_obs_lz.lazy(),on=["id_src","step","ships_sent"],how="left")
            .with_columns(pl.struct(["angle_min","angle_max","obstacle_list"])
                          .map_elements(IntervalProcessorPolars.compute_free_angles,return_dtype=pl.List(pl.List(pl.Float64))).alias("angle_list"))
            .filter(pl.col("angle_list").list.len()>0).collect()
        )

        # comet filter for polars
        awa_comets_lz=awa_lz.filter(pl.col("nature_src")=="comet")
        print(f"\nLZ comet sources: {sorted(awa_comets_lz['id_src'].unique().to_list()) if not awa_comets_lz.is_empty() else []}")
        awa_lz_filtered=awa_lz
        if not awa_comets_lz.is_empty():
            x_off=(awa_comets_lz["x_src"]-CENTER).abs().max();y_off=(awa_comets_lz["y_src"]-CENTER).abs().max()
            if max(x_off,y_off)>45:
                avoid=awa_comets_lz["id_src"].unique().to_list()
                print(f"LZ avoiding comet sources: {avoid}")
                awa_lz_filtered=awa_lz.filter(~pl.col("id_src").is_in(avoid))
        print(f"LZ awa after comet filter: {len(awa_lz_filtered)} rows")

        # polars top-5 AFTER comet filter (inside attacks chain)
        top5_lz_after_comet = (
            awa_lz_filtered.lazy().sort(["step","ships_sent"])
            .group_by(["id_src","id"],maintain_order=True).first()
            .sort(["step","ships_sent"])
            .group_by("id_src",maintain_order=True).head(5)
            .select(["id_src","id"]).collect()
        )
        top5_20_after_lz = top5_lz_after_comet.filter(pl.col("id_src")==20)["id"].to_list()
        print(f"\nLZ top-5 for id_src=20 (after comet filter, inside attacks chain): ids={sorted(top5_20_after_lz)}")

        # Now check: is planet 32 in the lz top-5 for id_src=20?
        print(f"\n*** Is planet 32 in LZ top-5? {'YES' if 32 in top5_20_after_lz else 'NO'}")
        print(f"*** Is planet 32 in PD top-5? {'YES' if 32 in top5_20_before else 'NO'}")

        # Build the full attacks for polars
        attacks_lz = (
            awa_lz_filtered.lazy()
            .sort(["step","ships_sent"])
            .group_by(["id_src","id"],maintain_order=True).first()
            .sort(["step","ships_sent"])
            .group_by("id_src",maintain_order=True).head(5)
            .select(["id_src","id"])
            .join(awa_lz_filtered.lazy(),on=["id_src","id"],how="left")
            .filter(pl.col("owner")!=player_id)
            .with_columns(pl.when(pl.col("owner")==-1).then(pl.col("ships")).otherwise(pl.col("ships")+pl.col("production")).alias("ships_needed"))
            .filter((pl.col("ships_needed")+1<=pl.col("ships_sent"))&(pl.col("ships_sent")<=pl.col("ships_needed")+pl.col("production_src")+1))
            .sort(["step","ships_sent"])
            .group_by(["id_src","id"],maintain_order=True).first()
            .with_columns((pl.col("ships_needed")/pl.col("production_src")).alias("time_cost"))
            .with_columns(pl.col("time_cost").sum().over("id_src").alias("total_time_cost"))
            .with_columns(((pl.col("total_time_cost")-pl.col("time_cost")-pl.col("step_diff"))*pl.col("production")).alias("score"))
            .sort("score",descending=True)
            .group_by("id_src",maintain_order=True).first()
            .filter(pl.col("ships_sent")<=pl.col("ships_min"))
            .collect()
        )
        print(f"\nLZ attacks (before final_angle):")
        print(attacks_lz.select(["id_src","id","ships_sent","step","ships_min","score"]).to_pandas().to_string())

        # Focused: check if id_src=20 is in attacks for both
        print(f"\nPD id_src=20 in final attacks: {20 in attacks_pd['id_src'].values}")
        print(f"LZ id_src=20 in final attacks: {20 in attacks_lz['id_src'].to_list()}")

        break

    # Use placeholder to step (we need the actual pd moves for game state)
    mine_s = (
        df.assign(is_mine=lambda d:(d["owner"]==player_id).astype(int)).groupby("id")
        .agg(step_src=("step","first"),x_src=("x","first"),y_src=("y","first"),radius_src=("radius","first"),
             ships_min=("ships","min"),production_src=("production","first"),nature_src=("nature","first"),
             owner_src=("owner","first"),row_count=("ships","size"),is_mine=("is_mine","sum"))
        .query("row_count == is_mine and owner_src==@player_id").reset_index(drop=False).rename(columns={"id":"id_src"})
    )
    expanded_s = (
        mine_s.assign(ships_sent=(mine_s["ships_min"]+mine_s["production_src"]*NB_STEPS_SIM).apply(lambda n:list(range(1,n+1))))
        .explode("ships_sent").astype({"ships_sent":int}).reset_index(drop=True)
    )
    df_st_s = expanded_s.merge(df,how="cross").query("step > step_src and id != id_src")
    pa_s = (
        df_st_s.assign(
            dist_tgt_src=lambda d:((d["x"]-d["x_src"])**2+(d["y"]-d["y_src"])**2)**0.5,
            step_diff=lambda d:d["step"]-d["step_src"],
            fleet_speed=lambda d:1.0+(MAX_SPEED-1.0)*(np.log(d["ships_sent"])/math.log(1000))**1.5,
            dist_fleet_src_min=lambda d:d["step_diff"]*d["fleet_speed"]+PLANET_MARGIN+d["radius_src"],
            dist_fleet_src_max=lambda d:(d["step_diff"]+1)*d["fleet_speed"]+PLANET_MARGIN+d["radius_src"],
            collision=lambda d:((d["dist_tgt_src"]-d["radius"]<d["dist_fleet_src_min"])&(d["dist_fleet_src_min"]<d["dist_tgt_src"]+d["radius"]))|((d["dist_tgt_src"]-d["radius"]<d["dist_fleet_src_max"])&(d["dist_fleet_src_max"]<d["dist_tgt_src"]+d["radius"])))
        .query("collision")
        .assign(crossing_sun=lambda d:d.apply(lambda row:point_to_segment_distance((CENTER,CENTER),(row["x_src"],row["y_src"]),(row["x"],row["y"]))<SUN_RADIUS+PLANET_MARGIN,axis=1).astype(bool))
        .query("not crossing_sun")
        .assign(
            angle=lambda d:np.arctan2(d["y"]-d["y_src"],d["x"]-d["x_src"]),
            radius_angle=lambda d:np.maximum(
                np.arccos(((d["dist_tgt_src"]**2+d["dist_fleet_src_min"]**2-d["radius"]**2)/(2*d["dist_tgt_src"]*d["dist_fleet_src_min"])).clip(-1,1)),
                np.arccos(((d["dist_tgt_src"]**2+d["dist_fleet_src_max"]**2-d["radius"]**2)/(2*d["dist_tgt_src"]*d["dist_fleet_src_max"])).clip(-1,1))
            ),
            angle_min=lambda d:np.mod(d["angle"]-d["radius_angle"],2*math.pi),
            angle_max=lambda d:np.mod(d["angle"]+d["radius_angle"],2*math.pi),
        ).sort_values("step",ascending=True)
    )
    if pa_s.empty:
        moves_pd_s = []
    else:
        df_obs_s=IntervalProcessor.create_cumulative_obstacles(pa_s)
        awa_s=(pa_s.merge(df_obs_s,how="left",on=["id_src","step","ships_sent"])
               .assign(angle_list=lambda d:d.apply(IntervalProcessor.compute_free_angles,axis=1))
               .query("angle_list.str.len() > 0"))
        top5_s=(awa_s.sort_values(["step","ships_sent"],ascending=True).groupby(["id_src","id"],as_index=False).first()
                .sort_values(["step","ships_sent"],ascending=True).groupby("id_src",as_index=False).head(5)[["id_src","id"]])
        awa_comets_s=awa_s.query("nature_src=='comet'");moves_pd_s=[]
        if not awa_comets_s.empty and max(max((awa_comets_s["x_src"]-CENTER).abs()),max((awa_comets_s["y_src"]-CENTER).abs()))>45:
            moves_pd_s+=(awa_comets_s.query("ships_sent<=ships_min").sort_values(["ships_sent","step"],ascending=[False,True])
                         .groupby("id_src",as_index=False).first()[["id_src","angle","ships_sent"]].values.tolist())
            id_to_avoid=awa_comets_s["id_src"].unique().tolist();awa_s=awa_s.query("id_src not in @id_to_avoid")
        atk_s=(top5_s.merge(awa_s,how="left",on=["id_src","id"]).query("@player_id != owner")
               .assign(ships_needed=lambda d:np.where(d["owner"]==-1,d["ships"],d["ships"]+d["production"]))
               .query("ships_needed + 1 <= ships_sent and ships_sent <= ships_needed + production_src + 1")
               .sort_values(["step","ships_sent"],ascending=True).groupby(["id_src","id"],as_index=False).first()
               .assign(time_cost=lambda d:d["ships_needed"]/d["production_src"],
                       total_time_cost=lambda d:d.groupby("id_src")["time_cost"].transform("sum"),
                       score=lambda d:(d["total_time_cost"]-d["time_cost"]-d["step_diff"])*d["production"])
               .sort_values("score",ascending=False).groupby("id_src",as_index=False).first()
               .query("ships_sent <= ships_min")
               .assign(final_angle=lambda d:IntervalProcessor.interval_to_final_angle(d["angle_list"])))
        moves_pd_s+=atk_s[["id_src","final_angle","ships_sent"]].values.tolist()
    rng_action=random_agent_fn(obs1)
    env.step([moves_pd_s,rng_action])
    if env.state[0].status!="ACTIVE":
        print(f"Game ended at step {env_step}"); break

print("Done")
